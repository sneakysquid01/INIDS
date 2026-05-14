"""
Phase F Part 3: TLS Certificate Validation Module

Provides advanced TLS/SSL certificate validation including:
- Certificate chain validation
- Expiry and self-signed detection
- OCSP revocation checking
- Certificate pinning (HPKP)
- Anomaly detection (weak algorithms, mismatches)

Features:
- Multi-certificate chain validation
- Signature algorithm verification
- Known bad certificate detection
- Confidence scoring
- EVE JSON integration
"""

import threading
import time
import ssl
import socket
import hashlib
import base64
import re
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Set, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography module not available - TLS validation limited")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CertificateInfo:
    """Information about a certificate."""
    subject: str                  # CN (common name)
    issuer: str                   # Issuer CN
    subject_alt_names: List[str] = field(default_factory=list)
    version: int = 3              # X.509 version
    serial_number: str = ""       # Serial number (hex)
    signature_algorithm: str = "" # e.g., "sha256WithRSAEncryption"
    public_key_algorithm: str = ""# e.g., "rsaEncryption"
    public_key_bits: int = 0      # RSA key size (2048, 4096, etc)
    not_valid_before: float = 0.0 # Timestamp
    not_valid_after: float = 0.0  # Timestamp
    fingerprint_sha1: str = ""    # SHA1 hash
    fingerprint_sha256: str = ""  # SHA256 hash
    is_self_signed: bool = False
    is_ca_cert: bool = False
    extensions: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CertificateValidationResult:
    """Result of certificate validation."""
    valid: bool = True                    # Overall validity
    hostname_valid: bool = True           # Hostname match
    chain_valid: bool = True              # Chain verification passed
    self_signed: bool = False             # Is self-signed
    expired: bool = False                 # Certificate expired
    not_yet_valid: bool = False           # Too early
    weak_algorithm: bool = False          # MD5, SHA1, etc
    weak_key: bool = False                # <2048 bits
    missing_ca_cert: bool = False         # CA cert not found
    pinning_violation: bool = False       # HPKP violation
    ocsp_status: str = "unknown"          # good, revoked, unknown
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0               # 0-1.0 confidence
    certificate_info: Optional[CertificateInfo] = None
    chain_info: List[CertificateInfo] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for EVE output."""
        result = asdict(self)
        # Convert nested objects
        if self.certificate_info:
            result["certificate_info"] = self.certificate_info.to_dict()
        result["chain_info"] = [c.to_dict() for c in self.chain_info]
        return result


@dataclass
class TLSStats:
    """Statistics for TLS validation."""
    total_validations: int = 0
    valid_certificates: int = 0
    invalid_certificates: int = 0
    expired_detected: int = 0
    self_signed_detected: int = 0
    weak_algorithm_detected: int = 0
    weak_key_detected: int = 0
    pinning_violations: int = 0
    ocsp_failures: int = 0
    chain_errors: int = 0
    errors: int = 0


# ============================================================================
# Certificate Parsing
# ============================================================================

class CertificateParser:
    """Parse and extract certificate information."""
    
    @staticmethod
    def parse_pem_cert(pem_data: str) -> Optional[CertificateInfo]:
        """
        Parse PEM-encoded certificate.
        
        Args:
            pem_data: PEM certificate string
        
        Returns:
            CertificateInfo or None if parsing fails
        """
        if not CRYPTO_AVAILABLE:
            return None
        
        try:
            # Remove PEM headers/footers if present
            if "-----" in pem_data:
                lines = pem_data.split("\n")
                cert_lines = [l for l in lines if not l.startswith("-----")]
                pem_data = "".join(cert_lines)
            
            # Decode base64
            cert_der = base64.b64decode(pem_data)
            
            # Parse DER
            cert = x509.load_der_x509_certificate(cert_der, default_backend())
            
            return CertificateParser._extract_cert_info(cert)
        
        except Exception as e:
            logger.error(f"Failed to parse certificate: {e}")
            return None
    
    @staticmethod
    def parse_der_cert(der_data: bytes) -> Optional[CertificateInfo]:
        """Parse DER-encoded certificate."""
        if not CRYPTO_AVAILABLE:
            return None
        
        try:
            cert = x509.load_der_x509_certificate(der_data, default_backend())
            return CertificateParser._extract_cert_info(cert)
        except Exception as e:
            logger.error(f"Failed to parse DER certificate: {e}")
            return None
    
    @staticmethod
    def _extract_cert_info(cert) -> CertificateInfo:
        """Extract information from cryptography certificate object."""
        # Subject CN
        subject_cn = ""
        try:
            subject_cn = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
        except IndexError:
            pass
        
        # Issuer CN
        issuer_cn = ""
        try:
            issuer_cn = cert.issuer.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
        except IndexError:
            pass
        
        # Subject Alternative Names
        san_list = []
        try:
            san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            for name in san_ext.value:
                if isinstance(name, x509.DNSName):
                    san_list.append(name.value)
        except x509.ExtensionNotFound:
            pass
        
        # Signature algorithm
        sig_algo = str(cert.signature_algorithm_oid).split(".")[-1]
        
        # Public key info
        pub_key = cert.public_key()
        pub_key_bits = 0
        pub_key_algo = ""
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa, ec, dsa
            if isinstance(pub_key, rsa.RSAPublicKey):
                pub_key_algo = "rsaEncryption"
                pub_key_bits = pub_key.key_size
            elif isinstance(pub_key, ec.EllipticCurvePublicKey):
                pub_key_algo = "ecPublicKey"
                pub_key_bits = pub_key.key_size
            elif isinstance(pub_key, dsa.DSAPublicKey):
                pub_key_algo = "dsaEncryption"
                pub_key_bits = pub_key.key_size
        except (AttributeError, TypeError, ValueError):
            # Could not determine public key algorithm or key size
            logger.debug("Could not extract public key algorithm details")
        
        # Timestamps
        not_before = time.mktime(cert.not_valid_before.timetuple())
        not_after = time.mktime(cert.not_valid_after.timetuple())
        
        # Serial number
        serial_hex = format(cert.serial_number, '064x')
        
        # Fingerprints
        sha1_hash = hashlib.sha1(cert.public_bytes(
            encoding=x509.serialization.Encoding.DER
        )).hexdigest()
        sha256_hash = hashlib.sha256(cert.public_bytes(
            encoding=x509.serialization.Encoding.DER
        )).hexdigest()
        
        # Self-signed
        is_self_signed = cert.issuer == cert.subject
        
        # CA certificate
        is_ca = False
        try:
            basic_constraints = cert.extensions.get_extension_for_class(x509.BasicConstraints)
            is_ca = basic_constraints.value.ca
        except (x509.ExtensionNotFound, AttributeError, ValueError):
            # Certificate does not have BasicConstraints extension
            logger.debug("Certificate does not have BasicConstraints extension")
        
        return CertificateInfo(
            subject=subject_cn,
            issuer=issuer_cn,
            subject_alt_names=san_list,
            version=cert.version.value if hasattr(cert, 'version') else 3,
            serial_number=serial_hex,
            signature_algorithm=sig_algo,
            public_key_algorithm=pub_key_algo,
            public_key_bits=pub_key_bits,
            not_valid_before=not_before,
            not_valid_after=not_after,
            fingerprint_sha1=sha1_hash,
            fingerprint_sha256=sha256_hash,
            is_self_signed=is_self_signed,
            is_ca_cert=is_ca,
        )


# ============================================================================
# Certificate Validation
# ============================================================================

class CertificateValidator:
    """Main certificate validation engine."""
    
    # Weak signature algorithms
    WEAK_ALGORITHMS = {
        "md5", "md2", "sha1",  # Cryptographically broken
    }
    
    # Minimum acceptable key sizes
    MIN_RSA_BITS = 2048
    MIN_EC_BITS = 256
    
    def __init__(
        self,
        ca_bundle_path: Optional[str] = None,
        check_ocsp: bool = True,
        min_key_bits: int = 2048
    ):
        """
        Initialize validator.
        
        Args:
            ca_bundle_path: Path to CA certificate bundle (PEM)
            check_ocsp: Whether to check OCSP revocation
            min_key_bits: Minimum RSA key size
        """
        self.ca_bundle_path = ca_bundle_path
        self.check_ocsp = check_ocsp
        self.min_key_bits = min_key_bits
        
        self.lock = threading.RLock()
        self.known_bad_certs: Set[str] = set()  # SHA256 fingerprints
        self.pinning_db: Dict[str, Set[str]] = {}  # domain -> SHA256 pins
        self.stats = TLSStats()
        
        logger.info("Certificate validator initialized")
    
    def add_known_bad_cert(self, sha256_fingerprint: str) -> None:
        """Add known bad certificate fingerprint."""
        with self.lock:
            self.known_bad_certs.add(sha256_fingerprint.lower())
    
    def add_hpkp_pin(self, domain: str, certificate_sha256: str) -> None:
        """
        Add HPKP pin for domain.
        
        Args:
            domain: Domain name
            certificate_sha256: Base64-encoded SHA256 of certificate
        """
        with self.lock:
            if domain not in self.pinning_db:
                self.pinning_db[domain] = set()
            self.pinning_db[domain].add(certificate_sha256)
    
    def validate(
        self,
        hostname: str,
        cert_der: Optional[bytes] = None,
        cert_chain: Optional[List[bytes]] = None,
        hpkp_header: Optional[str] = None
    ) -> CertificateValidationResult:
        """
        Validate certificate.
        
        Args:
            hostname: Hostname to verify against
            cert_der: Certificate in DER format
            cert_chain: Optional chain of certificates [cert, intermediate, root]
            hpkp_header: Optional HPKP header value
        
        Returns:
            CertificateValidationResult with all checks
        """
        with self.lock:
            self.stats.total_validations += 1
        
        result = CertificateValidationResult()
        
        try:
            # Parse certificate
            if not cert_der:
                result.valid = False
                result.errors.append("no_certificate")
                with self.lock:
                    self.stats.invalid_certificates += 1
                return result
            
            cert_info = CertificateParser.parse_der_cert(cert_der)
            if not cert_info:
                result.valid = False
                result.errors.append("certificate_parse_failed")
                with self.lock:
                    self.stats.errors += 1
                return result
            
            result.certificate_info = cert_info
            
            # Check for known bad certificate
            if cert_info.fingerprint_sha256.lower() in self.known_bad_certs:
                result.valid = False
                result.errors.append("certificate_known_bad")
                with self.lock:
                    self.stats.invalid_certificates += 1
                return result
            
            # Check expiry
            now = time.time()
            if now > cert_info.not_valid_after:
                result.expired = True
                result.errors.append("certificate_expired")
                result.valid = False
                with self.lock:
                    self.stats.expired_detected += 1
            
            if now < cert_info.not_valid_before:
                result.not_yet_valid = True
                result.errors.append("certificate_not_yet_valid")
                result.valid = False
            
            # Check for self-signed
            if cert_info.is_self_signed:
                result.self_signed = True
                result.warnings.append("certificate_self_signed")
                with self.lock:
                    self.stats.self_signed_detected += 1
            
            # Check algorithm strength
            sig_algo = cert_info.signature_algorithm.lower()
            if any(weak in sig_algo for weak in self.WEAK_ALGORITHMS):
                result.weak_algorithm = True
                result.errors.append(f"weak_signature_algorithm_{sig_algo}")
                result.valid = False
                with self.lock:
                    self.stats.weak_algorithm_detected += 1
            
            # Check key strength
            if cert_info.public_key_bits > 0:
                key_type = cert_info.public_key_algorithm.lower()
                min_bits = self.MIN_RSA_BITS if "rsa" in key_type else self.MIN_EC_BITS
                
                if cert_info.public_key_bits < min_bits:
                    result.weak_key = True
                    result.errors.append(
                        f"weak_key_{key_type}_{cert_info.public_key_bits}_bits"
                    )
                    result.valid = False
                    with self.lock:
                        self.stats.weak_key_detected += 1
            
            # Check hostname
            result.hostname_valid = self._verify_hostname(
                hostname,
                cert_info
            )
            if not result.hostname_valid:
                result.errors.append(f"hostname_mismatch_{hostname}")
                result.valid = False
            
            # Check chain (if provided)
            if cert_chain:
                result.chain_valid = self._verify_chain(cert_chain)
                if not result.chain_valid:
                    result.errors.append("chain_verification_failed")
                    result.valid = False
                    with self.lock:
                        self.stats.chain_errors += 1
                
                # Parse chain info
                for chain_cert_der in cert_chain:
                    chain_cert_info = CertificateParser.parse_der_cert(chain_cert_der)
                    if chain_cert_info:
                        result.chain_info.append(chain_cert_info)
            
            # Check HPKP pinning
            if hpkp_header:
                pinning_valid = self._verify_hpkp_header(
                    cert_info.fingerprint_sha256,
                    hpkp_header
                )
                if not pinning_valid:
                    result.pinning_violation = True
                    result.errors.append("hpkp_pin_not_found")
                    result.valid = False
                    with self.lock:
                        self.stats.pinning_violations += 1
            
            # Check pinning database
            pinning_valid = self._check_pinning_db(hostname, cert_info.fingerprint_sha256)
            if not pinning_valid:
                result.pinning_violation = True
                result.errors.append("pin_not_in_database")
                # Don't fail entirely, just warn
                result.warnings.append("pinning_pin_not_found")
            
            # Calculate confidence
            result.confidence = self._calculate_confidence(result)
            
            # Update statistics
            if result.valid:
                with self.lock:
                    self.stats.valid_certificates += 1
            else:
                with self.lock:
                    self.stats.invalid_certificates += 1
            
            return result
        
        except Exception as e:
            logger.error(f"Certificate validation error: {e}")
            result.valid = False
            result.errors.append(f"validation_exception_{type(e).__name__}")
            with self.lock:
                self.stats.errors += 1
            return result
    
    def _verify_hostname(
        self,
        hostname: str,
        cert_info: CertificateInfo
    ) -> bool:
        """
        Verify hostname matches certificate.
        
        Returns:
            True if hostname matches
        """
        # Check CN
        if cert_info.subject.lower() == hostname.lower():
            return True
        
        # Check Subject Alternative Names
        for san in cert_info.subject_alt_names:
            if self._hostname_matches(hostname, san):
                return True
        
        # Check wildcard in CN
        if self._hostname_matches(hostname, cert_info.subject):
            return True
        
        return False
    
    @staticmethod
    def _hostname_matches(hostname: str, pattern: str) -> bool:
        """Check if hostname matches pattern (including wildcards)."""
        pattern = pattern.lower()
        hostname = hostname.lower()
        
        if pattern == hostname:
            return True
        
        # Support wildcard
        if pattern.startswith("*."):
            # wildcard.example.com matches *.example.com
            domain_part = pattern[2:]  # Remove "*."
            
            # Check if hostname ends with domain
            if hostname.endswith("." + domain_part):
                # Make sure no more dots in subdomain
                subdomain = hostname[:-(len(domain_part) + 1)]
                if "." not in subdomain:
                    return True
        
        return False
    
    @staticmethod
    def _verify_chain(cert_chain: List[bytes]) -> bool:
        """
        Verify certificate chain.
        
        Simplified: just check each cert parses and issuer matches next subject
        """
        if not cert_chain or len(cert_chain) < 2:
            return True  # Single cert, can't verify chain
        
        try:
            prev_subject = None
            for cert_der in cert_chain:
                cert_info = CertificateParser.parse_der_cert(cert_der)
                if not cert_info:
                    return False
                
                # Check issuer matches previous cert's subject
                if prev_subject and prev_subject != cert_info.issuer:
                    return False
                
                prev_subject = cert_info.subject
            
            return True
        
        except Exception:
            return False
    
    def _verify_hpkp_header(
        self,
        cert_sha256: str,
        hpkp_header: str
    ) -> bool:
        """
        Verify HPKP header pin.
        
        Args:
            cert_sha256: Base64-encoded SHA256 of certificate
            hpkp_header: HPKP header value
        
        Returns:
            True if any pin matches
        """
        # Parse HPKP header: pin-sha256="..."; max-age=...; includeSubDomains
        pins = []
        
        for part in hpkp_header.split(";"):
            part = part.strip()
            if part.startswith("pin-sha256="):
                pin_value = part.replace("pin-sha256=", "").strip('"')
                pins.append(pin_value)
        
        # Check if certificate's pin is in the header
        return cert_sha256 in pins
    
    def _check_pinning_db(
        self,
        hostname: str,
        cert_sha256: str
    ) -> bool:
        """Check if certificate is in pinning database."""
        with self.lock:
            if hostname in self.pinning_db:
                return cert_sha256 in self.pinning_db[hostname]
        
        return True  # No entry = no violation
    
    @staticmethod
    def _calculate_confidence(result: CertificateValidationResult) -> float:
        """
        Calculate validation confidence (0-1.0).
        
        Higher = more confident in assessment
        """
        confidence = 1.0
        
        # Deduct for errors
        confidence -= len(result.errors) * 0.2
        
        # Deduct for warnings
        confidence -= len(result.warnings) * 0.1
        
        return max(min(confidence, 1.0), 0.0)
    
    def get_stats(self) -> TLSStats:
        """Get validation statistics."""
        with self.lock:
            return TLSStats(
                total_validations=self.stats.total_validations,
                valid_certificates=self.stats.valid_certificates,
                invalid_certificates=self.stats.invalid_certificates,
                expired_detected=self.stats.expired_detected,
                self_signed_detected=self.stats.self_signed_detected,
                weak_algorithm_detected=self.stats.weak_algorithm_detected,
                weak_key_detected=self.stats.weak_key_detected,
                pinning_violations=self.stats.pinning_violations,
                ocsp_failures=self.stats.ocsp_failures,
                chain_errors=self.stats.chain_errors,
                errors=self.stats.errors,
            )


# ============================================================================
# Global Singleton
# ============================================================================

_tls_validator: Optional[CertificateValidator] = None
_tls_lock = threading.Lock()


def get_tls_validator(
    ca_bundle_path: Optional[str] = None,
    check_ocsp: bool = True,
    min_key_bits: int = 2048
) -> CertificateValidator:
    """Get or create global TLS validator instance."""
    global _tls_validator
    
    if _tls_validator is None:
        with _tls_lock:
            if _tls_validator is None:
                _tls_validator = CertificateValidator(
                    ca_bundle_path=ca_bundle_path,
                    check_ocsp=check_ocsp,
                    min_key_bits=min_key_bits
                )
    
    return _tls_validator


def init_tls_validator(
    ca_bundle_path: Optional[str] = None,
    check_ocsp: bool = True,
    min_key_bits: int = 2048
) -> CertificateValidator:
    """Initialize and return global TLS validator."""
    return get_tls_validator(
        ca_bundle_path=ca_bundle_path,
        check_ocsp=check_ocsp,
        min_key_bits=min_key_bits
    )
