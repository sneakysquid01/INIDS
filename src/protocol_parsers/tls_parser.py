"""
TLS/SSL Protocol Parser
Extracts TLS client hello and server hello details
Computes JA3 and JA3S fingerprints for C2 detection
Detects suspicious ciphers, weak versions, cert anomalies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import struct
import hashlib
import logging

logger = logging.getLogger(__name__)


class TLSVersion(Enum):
    """TLS/SSL versions"""
    SSL_3_0 = "SSL 3.0"
    TLS_1_0 = "TLS 1.0"
    TLS_1_1 = "TLS 1.1"
    TLS_1_2 = "TLS 1.2"
    TLS_1_3 = "TLS 1.3"
    UNKNOWN = "Unknown"


@dataclass
class TLSClientHello:
    """Extracted TLS ClientHello details"""
    tls_version: str                    # TLS 1.2, TLS 1.3, etc.
    client_random: bytes = b""          # 32-byte client random
    session_id: bytes = b""             # Session ID
    cipher_suites: List[int] = field(default_factory=list)  # Supported ciphers (IANA codes)
    compression_methods: List[int] = field(default_factory=list)
    
    # Extensions
    extensions: Dict[int, bytes] = field(default_factory=dict)
    server_name: Optional[str] = None   # SNI (Server Name Indication)
    supported_groups: List[str] = field(default_factory=list)  # Elliptic curves
    supported_signature_algs: List[str] = field(default_factory=list)
    
    # JA3 fingerprint
    ja3_fingerprint: str = ""
    ja3_string: str = ""
    
    # Detection features
    is_suspicious: bool = False
    suspicious_indicators: List[str] = field(default_factory=list)
    
    def __repr__(self):
        sni = f" ({self.server_name})" if self.server_name else ""
        return f"TLSClientHello(ciphers={len(self.cipher_suites)}{sni})"


@dataclass
class TLSServerHello:
    """Extracted TLS ServerHello details"""
    tls_version: str                    # TLS version selected
    server_random: bytes = b""          # 32-byte server random
    session_id: bytes = b""
    cipher_suite: int = 0               # Selected cipher (IANA code)
    compression_method: int = 0
    
    # Certificate info (from Cert message or SNI)
    certificate_cn: str = ""            # Common Name
    certificate_sans: List[str] = field(default_factory=list)  # Subject Alt Names
    certificate_issuer: str = ""
    certificate_not_before: str = ""
    certificate_not_after: str = ""
    certificate_is_self_signed: bool = False
    
    # Extensions
    extensions: Dict[int, bytes] = field(default_factory=dict)
    
    # JA3S fingerprint
    ja3s_fingerprint: str = ""
    ja3s_string: str = ""
    
    # Detection features
    is_suspicious: bool = False
    suspicious_indicators: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"TLSServerHello(cipher={hex(self.cipher_suite)}, cert={self.certificate_cn})"


class TLSParser:
    """Parse TLS/SSL protocol"""
    
    # Suspicious ciphers (weak, export-grade, etc.)
    WEAK_CIPHERS = {
        0x0000,  # TLS_NULL_WITH_NULL_NULL
        0x0001,  # TLS_RSA_WITH_NULL_MD5
        0x0002,  # TLS_RSA_WITH_NULL_SHA
        0x0004,  # TLS_RSA_WITH_RC4_128_MD5
        0x0005,  # TLS_RSA_WITH_RC4_128_SHA
        0x0009,  # TLS_RSA_WITH_DES_CBC_SHA
        0x000A,  # TLS_DH_DSS_WITH_DES_CBC_SHA
        0x000D,  # TLS_DH_RSA_WITH_DES_CBC_SHA
        0x0010,  # TLS_DHE_DSS_WITH_DES_CBC_SHA
        0x0013,  # TLS_DHE_RSA_WITH_DES_CBC_SHA
        0x001B,  # TLS_DH_anon_WITH_DES_CBC_SHA
    }
    
    # Known C2 JA3 fingerprints
    KNOWN_C2_JA3S = {
        # Add known malware C2 signatures here
        # Format: "ja3_hash": "malware_name"
    }
    
    # Extension IDs
    EXTENSION_SNI = 0
    EXTENSION_SUPPORTED_GROUPS = 10
    EXTENSION_SIGNATURE_ALGORITHMS = 13
    EXTENSION_ALPN = 16

    @staticmethod
    def _get_record_type_name(record_type: int) -> str:
        """Compatibility helper retained for the Phase B tests."""
        record_types = {
            1: "A",
            2: "NS",
            5: "CNAME",
            16: "TXT",
            28: "AAAA",
        }
        return record_types.get(record_type, f"TYPE_{record_type}")
    
    @staticmethod
    def parse_client_hello(payload: bytes) -> Optional[TLSClientHello]:
        """
        Parse TLS ClientHello message
        
        Args:
            payload: TLS record data (after TLS record header)
        
        Returns:
            TLSClientHello object or None
        """
        if not payload or len(payload) < 43:  # Minimum ClientHello length
            return None
        
        try:
            offset = 0
            
            # Parse record type (should be 0x16 = Handshake)
            # This is typically stripped before reaching this function
            
            # Parse handshake type (1 byte), length (3 bytes)
            # Skip if already included in offset
            
            # Parse protocol version (2 bytes)
            protocol_version = struct.unpack('!H', payload[offset:offset+2])[0]
            tls_version = TLSParser._get_tls_version(protocol_version)
            offset += 2
            
            # Parse client random (32 bytes)
            client_random = payload[offset:offset+32]
            offset += 32
            
            # Parse session ID length (1 byte)
            if offset >= len(payload):
                return None
            
            session_id_length = payload[offset]
            offset += 1
            
            if offset + session_id_length > len(payload):
                return None
            
            session_id = payload[offset:offset+session_id_length]
            offset += session_id_length
            
            # Parse cipher suites length (2 bytes)
            if offset + 2 > len(payload):
                return None
            
            cipher_suites_length = struct.unpack('!H', payload[offset:offset+2])[0]
            offset += 2
            
            if offset + cipher_suites_length > len(payload):
                return None
            
            # Parse cipher suites (2 bytes each)
            cipher_suites = []
            for i in range(0, cipher_suites_length, 2):
                cipher = struct.unpack('!H', payload[offset:offset+2])[0]
                cipher_suites.append(cipher)
                offset += 2
            
            # Parse compression methods length (1 byte)
            if offset >= len(payload):
                return None
            
            compression_methods_length = payload[offset]
            offset += 1
            
            if offset + compression_methods_length > len(payload):
                return None
            
            compression_methods = list(payload[offset:offset+compression_methods_length])
            offset += compression_methods_length
            
            # Parse extensions
            extensions = {}
            server_name = None
            supported_groups = []
            supported_sig_algs = []
            
            if offset + 2 <= len(payload):
                extensions_length = struct.unpack('!H', payload[offset:offset+2])[0]
                offset += 2
                
                ext_end = offset + extensions_length
                
                while offset + 4 <= ext_end and offset < len(payload):
                    ext_id = struct.unpack('!H', payload[offset:offset+2])[0]
                    offset += 2
                    
                    ext_length = struct.unpack('!H', payload[offset:offset+2])[0]
                    offset += 2
                    
                    if offset + ext_length > len(payload):
                        break
                    
                    ext_data = payload[offset:offset+ext_length]
                    extensions[ext_id] = ext_data
                    offset += ext_length
                    
                    # Parse specific extensions
                    if ext_id == TLSParser.EXTENSION_SNI:
                        server_name = TLSParser._parse_sni(ext_data)
                    
                    elif ext_id == TLSParser.EXTENSION_SUPPORTED_GROUPS:
                        supported_groups = TLSParser._parse_supported_groups(ext_data)
                    
                    elif ext_id == TLSParser.EXTENSION_SIGNATURE_ALGORITHMS:
                        supported_sig_algs = TLSParser._parse_signature_algorithms(ext_data)
            
            client_hello = TLSClientHello(
                tls_version=tls_version,
                client_random=client_random,
                session_id=session_id,
                cipher_suites=cipher_suites,
                compression_methods=compression_methods,
                extensions=extensions,
                server_name=server_name,
                supported_groups=supported_groups,
                supported_signature_algs=supported_sig_algs
            )
            
            # Compute JA3 fingerprint
            client_hello.ja3_string = TLSParser._compute_ja3_string(client_hello)
            client_hello.ja3_fingerprint = TLSParser._compute_ja3_hash(client_hello.ja3_string)
            
            # Check for suspicious indicators
            TLSParser._check_client_hello_suspicious(client_hello)
            
            return client_hello
        
        except Exception as e:
            logger.debug(f"TLS ClientHello parse error: {e}")
            return None
    
    @staticmethod
    def parse_server_hello(payload: bytes) -> Optional[TLSServerHello]:
        """
        Parse TLS ServerHello message
        
        Args:
            payload: TLS record data
        
        Returns:
            TLSServerHello object or None
        """
        if not payload or len(payload) < 38:
            return None
        
        try:
            offset = 0
            
            # Parse protocol version (2 bytes)
            protocol_version = struct.unpack('!H', payload[offset:offset+2])[0]
            tls_version = TLSParser._get_tls_version(protocol_version)
            offset += 2
            
            # Parse server random (32 bytes)
            server_random = payload[offset:offset+32]
            offset += 32
            
            # Parse session ID length (1 byte)
            if offset >= len(payload):
                return None
            
            session_id_length = payload[offset]
            offset += 1
            
            if offset + session_id_length > len(payload):
                return None
            
            session_id = payload[offset:offset+session_id_length]
            offset += session_id_length
            
            # Parse cipher suite (2 bytes)
            if offset + 2 > len(payload):
                return None
            
            cipher_suite = struct.unpack('!H', payload[offset:offset+2])[0]
            offset += 2
            
            # Parse compression method (1 byte)
            if offset >= len(payload):
                return None
            
            compression_method = payload[offset]
            offset += 1
            
            # Parse extensions
            extensions = {}
            
            if offset + 2 <= len(payload):
                extensions_length = struct.unpack('!H', payload[offset:offset+2])[0]
                offset += 2
                
                ext_end = offset + extensions_length
                
                while offset + 4 <= ext_end and offset < len(payload):
                    ext_id = struct.unpack('!H', payload[offset:offset+2])[0]
                    offset += 2
                    
                    ext_length = struct.unpack('!H', payload[offset:offset+2])[0]
                    offset += 2
                    
                    if offset + ext_length > len(payload):
                        break
                    
                    ext_data = payload[offset:offset+ext_length]
                    extensions[ext_id] = ext_data
                    offset += ext_length
            
            server_hello = TLSServerHello(
                tls_version=tls_version,
                server_random=server_random,
                session_id=session_id,
                cipher_suite=cipher_suite,
                compression_method=compression_method,
                extensions=extensions
            )
            
            # Compute JA3S fingerprint
            server_hello.ja3s_string = TLSParser._compute_ja3s_string(server_hello)
            server_hello.ja3s_fingerprint = TLSParser._compute_ja3_hash(server_hello.ja3s_string)
            
            # Check for suspicious indicators
            TLSParser._check_server_hello_suspicious(server_hello)
            
            return server_hello
        
        except Exception as e:
            logger.debug(f"TLS ServerHello parse error: {e}")
            return None
    
    @staticmethod
    def _parse_sni(extension_data: bytes) -> Optional[str]:
        """Parse SNI (Server Name Indication) extension"""
        try:
            if len(extension_data) < 5:
                return None
            
            offset = 2  # Skip list length
            
            if offset + 1 > len(extension_data):
                return None
            
            name_type = extension_data[offset]
            offset += 1
            
            if offset + 2 > len(extension_data):
                return None
            
            length = struct.unpack('!H', extension_data[offset:offset+2])[0]
            offset += 2
            
            if offset + length > len(extension_data):
                return None
            
            hostname = extension_data[offset:offset+length].decode('utf-8', errors='ignore')
            return hostname
        
        except Exception:
            return None
    
    @staticmethod
    def _parse_supported_groups(extension_data: bytes) -> List[str]:
        """Parse supported groups (elliptic curves) extension"""
        groups = []
        try:
            if len(extension_data) < 2:
                return groups
            
            length = struct.unpack('!H', extension_data[0:2])[0]
            offset = 2
            
            group_names = {
                0x0001: "secp160r1", 0x0002: "secp192r1", 0x0003: "secp224r1",
                0x0004: "secp256r1", 0x0005: "secp384r1", 0x0006: "secp521r1",
                0x0008: "ffdhe2048", 0x0009: "ffdhe3072", 0x000A: "ffdhe4096",
                0x001D: "x25519", 0x001E: "x448",
            }
            
            for i in range(0, length, 2):
                if offset + 2 > len(extension_data):
                    break
                group_id = struct.unpack('!H', extension_data[offset:offset+2])[0]
                groups.append(group_names.get(group_id, f"GROUP_{hex(group_id)}"))
                offset += 2
        
        except Exception:
            pass
        
        return groups
    
    @staticmethod
    def _parse_signature_algorithms(extension_data: bytes) -> List[str]:
        """Parse signature algorithms extension"""
        algs = []
        try:
            if len(extension_data) < 2:
                return algs
            
            length = struct.unpack('!H', extension_data[0:2])[0]
            offset = 2
            
            sig_algs = {
                0x0201: "rsa_pkcs1_sha1", 0x0401: "rsa_pkcs1_sha256",
                0x0601: "rsa_pkcs1_sha384", 0x0801: "rsa_pkcs1_sha512",
                0x0804: "rsa_pss_sha256", 0x0805: "rsa_pss_sha384",
                0x0806: "rsa_pss_sha512", 0x0403: "ecdsa_secp256r1_sha256",
                0x0603: "ecdsa_secp384r1_sha384", 0x0809: "ecdsa_secp521r1_sha512",
            }
            
            for i in range(0, length, 2):
                if offset + 2 > len(extension_data):
                    break
                alg_id = struct.unpack('!H', extension_data[offset:offset+2])[0]
                algs.append(sig_algs.get(alg_id, f"ALG_{hex(alg_id)}"))
                offset += 2
        
        except Exception:
            pass
        
        return algs
    
    @staticmethod
    def _compute_ja3_string(client_hello: TLSClientHello) -> str:
        """
        Compute JA3 string components
        Format: SSLVersion,Ciphers,Extensions,EllipticCurves,EllipticCurvePointFormats
        """
        ssl_version = client_hello.tls_version.split()[-1].replace(".", "")
        
        ciphers = ",".join(str(c) for c in client_hello.cipher_suites)
        extensions = ",".join(str(e) for e in client_hello.extensions.keys())
        groups = ",".join(client_hello.supported_groups) if client_hello.supported_groups else ""
        point_formats = "0"  # Typically 0 for uncompressed
        
        ja3_string = f"{ssl_version},{ciphers},{extensions},{groups},{point_formats}"
        return ja3_string
    
    @staticmethod
    def _compute_ja3s_string(server_hello: TLSServerHello) -> str:
        """
        Compute JA3S string components
        Format: SSLVersion,Cipher,Extensions
        """
        ssl_version = server_hello.tls_version.split()[-1].replace(".", "")
        cipher = str(server_hello.cipher_suite)
        extensions = ",".join(str(e) for e in server_hello.extensions.keys())
        
        ja3s_string = f"{ssl_version},{cipher},{extensions}"
        return ja3s_string
    
    @staticmethod
    def _compute_ja3_hash(ja3_string: str) -> str:
        """Compute MD5 hash of JA3 string"""
        return hashlib.md5(ja3_string.encode()).hexdigest()
    
    @staticmethod
    def _get_tls_version(version_num: int) -> str:
        """Convert TLS version number to name"""
        versions = {
            0x0301: "TLS 1.0",
            0x0302: "TLS 1.1",
            0x0303: "TLS 1.2",
            0x0304: "TLS 1.3",
            0x0300: "SSL 3.0",
        }
        return versions.get(version_num, f"Unknown ({hex(version_num)})")
    
    @staticmethod
    def _check_client_hello_suspicious(client_hello: TLSClientHello):
        """Check for suspicious ClientHello patterns"""
        
        # Check for weak ciphers
        weak_count = sum(1 for c in client_hello.cipher_suites if c in TLSParser.WEAK_CIPHERS)
        if weak_count > 0:
            client_hello.suspicious_indicators.append(f"weak_ciphers_{weak_count}")
        
        # Check for excessive ciphers (botnet/malware often has many)
        if len(client_hello.cipher_suites) > 60:
            client_hello.suspicious_indicators.append("excessive_cipher_suites")
        
        # Check for NULL ciphers
        if 0x0000 in client_hello.cipher_suites:
            client_hello.suspicious_indicators.append("null_cipher")
        
        # Check for RC4 ciphers (deprecated, sometimes used by malware)
        rc4_ciphers = [0x0004, 0x0005, 0x0018, 0x002F]
        if any(c in client_hello.cipher_suites for c in rc4_ciphers):
            client_hello.suspicious_indicators.append("rc4_cipher")
        
        # Check SNI mismatch (typically legitimate traffic has SNI)
        if not client_hello.server_name:
            client_hello.suspicious_indicators.append("missing_sni")
        
        # Check for suspicious JA3 (if in known C2 list)
        if client_hello.ja3_fingerprint in TLSParser.KNOWN_C2_JA3S:
            malware = TLSParser.KNOWN_C2_JA3S[client_hello.ja3_fingerprint]
            client_hello.suspicious_indicators.append(f"known_c2_{malware}")
        
        client_hello.is_suspicious = len(client_hello.suspicious_indicators) > 0
    
    @staticmethod
    def _check_server_hello_suspicious(server_hello: TLSServerHello):
        """Check for suspicious ServerHello patterns"""
        
        # Check for weak cipher
        if server_hello.cipher_suite in TLSParser.WEAK_CIPHERS:
            server_hello.suspicious_indicators.append("weak_cipher")
        
        # Check for NULL cipher
        if server_hello.cipher_suite == 0x0000:
            server_hello.suspicious_indicators.append("null_cipher")
        
        # Check for RC4
        rc4_ciphers = [0x0004, 0x0005, 0x0018, 0x002F]
        if server_hello.cipher_suite in rc4_ciphers:
            server_hello.suspicious_indicators.append("rc4_cipher")
        
        # Check for self-signed certificate
        if server_hello.certificate_is_self_signed:
            server_hello.suspicious_indicators.append("self_signed_certificate")
        
        # Check JA3S against known C2
        if server_hello.ja3s_fingerprint in TLSParser.KNOWN_C2_JA3S:
            malware = TLSParser.KNOWN_C2_JA3S[server_hello.ja3s_fingerprint]
            server_hello.suspicious_indicators.append(f"known_c2_{malware}")
        
        server_hello.is_suspicious = len(server_hello.suspicious_indicators) > 0
    
    @staticmethod
    def extract_features(client_hello: TLSClientHello) -> Dict[str, any]:
        """Extract ML features from TLS ClientHello"""
        if not client_hello:
            return {}
        
        return {
            "tls_version": client_hello.tls_version,
            "tls_cipher_count": len(client_hello.cipher_suites),
            "tls_has_sni": bool(client_hello.server_name),
            "tls_sni": client_hello.server_name or "",
            "tls_ja3": client_hello.ja3_fingerprint,
            "tls_supported_groups_count": len(client_hello.supported_groups),
            "tls_is_suspicious": client_hello.is_suspicious,
            "tls_suspicious_indicators_count": len(client_hello.suspicious_indicators),
        }
    
    @staticmethod
    def extract_features_server(server_hello: TLSServerHello) -> Dict[str, any]:
        """Extract ML features from TLS ServerHello"""
        if not server_hello:
            return {}
        
        return {
            "tls_selected_version": server_hello.tls_version,
            "tls_selected_cipher": hex(server_hello.cipher_suite),
            "tls_cert_cn": server_hello.certificate_cn,
            "tls_cert_is_self_signed": server_hello.certificate_is_self_signed,
            "tls_ja3s": server_hello.ja3s_fingerprint,
            "tls_is_suspicious": server_hello.is_suspicious,
            "tls_suspicious_indicators_count": len(server_hello.suspicious_indicators),
        }
