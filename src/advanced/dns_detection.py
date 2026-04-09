"""
Phase F Part 2: DNS Detection Module

Provides advanced DNS attack detection including:
- Sinkhole redirection detection
- Domain Generation Algorithm (DGA) detection
- DNS tunneling detection
- DNS policy enforcement with RPZ rules

Features:
- Entropy analysis for DGA detection
- Domain length and pattern analysis
- Known sinkhole IP lists
- RPZ-style policy rules
- Cache for known domains
"""

import threading
import time
import math
import re
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Set, List, Tuple, Callable
from collections import Counter, OrderedDict
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DNSAnalysisResult:
    """Result of DNS query analysis."""
    domain: str                    # Queried domain
    query_type: str               # A, AAAA, MX, etc.
    response_ip: Optional[str]    # Response IP (if any)
    is_sinkhole: bool = False     # Likely sinkhole redirect
    sinkhole_reason: str = ""     # Why it's a sinkhole
    dga_score: float = 0.0        # 0-5 scale, >4.0 is DGA
    dga_indicators: List[str] = field(default_factory=list)
    is_tunneling: bool = False    # DNS tunneling detected
    tunneling_score: float = 0.0  # 0-1 scale
    tunneling_indicators: List[str] = field(default_factory=list)
    policy_violations: List[str] = field(default_factory=list)
    anomaly_score: float = 0.0    # 0-5 scale, overall anomaly
    query_rate: float = 0.0       # Queries/sec from this source
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for EVE output."""
        return asdict(self)
    
    @property
    def is_anomalous(self) -> bool:
        """Check if overall anomalous."""
        return (self.is_sinkhole or 
                self.dga_score > 4.0 or 
                self.is_tunneling or 
                len(self.policy_violations) > 0)


@dataclass
class DNSStats:
    """Statistics for DNS detection."""
    total_queries: int = 0
    sinkhole_detected: int = 0
    dga_detected: int = 0
    tunneling_detected: int = 0
    policy_violations: int = 0
    anomalies: int = 0
    errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


# ============================================================================
# Entropy Calculation
# ============================================================================

def calculate_entropy(data: str) -> float:
    """
    Calculate Shannon entropy of string.
    
    High entropy = more random, likely DGA
    Low entropy = more patterns, likely legitimate
    
    Entropy scale: 0-4.7 for domains (base-26 alphanumeric)
    - 2.0-3.0: English-like (legitimate)
    - 3.0-4.0: Random-like (suspicious)
    - 4.0+: Very random (likely DGA)
    """
    if not data:
        return 0.0
    
    # Count character frequencies
    counter = Counter(data.lower())
    total = len(data)
    
    # Calculate entropy
    entropy = 0.0
    for count in counter.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    
    return entropy


def calculate_domain_entropy(domain: str) -> float:
    """Calculate entropy of domain name (excluding TLD)."""
    # Remove TLD
    parts = domain.split(".")
    if len(parts) > 1:
        name = parts[0]  # Just first label
    else:
        name = domain
    
    return calculate_entropy(name)


# ============================================================================
# Domain Pattern Analysis
# ============================================================================

def analyze_domain_patterns(domain: str) -> Tuple[float, List[str]]:
    """
    Analyze domain for suspicious patterns.
    
    Returns (suspicion_score, indicators)
    Score: 0-5
    """
    score = 0.0
    indicators = []
    
    # Remove TLD
    parts = domain.split(".")
    domain_name = parts[0] if parts else domain
    
    # Check length (DGA often uses similar lengths)
    if len(domain_name) > 20:
        score += 0.5
        indicators.append("long_domain_name")
    
    # Check for numbers (DGA often mixes letters/numbers)
    if any(c.isdigit() for c in domain_name):
        score += 0.2
        indicators.append("contains_numbers")
    
    # Check for dictionary words (legitimate)
    common_words = {"mail", "smtp", "www", "web", "api", "data", "cloud", "app"}
    if domain_name.lower() in common_words:
        score = max(0, score - 0.5)
        indicators.append("common_word")
    
    # Check for repeating characters
    if any(domain_name.lower().count(c) > 3 for c in set(domain_name.lower())):
        score += 0.3
        indicators.append("repeating_characters")
    
    # Check for vowel-consonant patterns
    vowels = "aeiou"
    vowel_count = sum(1 for c in domain_name.lower() if c in vowels)
    vowel_ratio = vowel_count / len(domain_name) if domain_name else 0
    
    if vowel_ratio < 0.1 or vowel_ratio > 0.7:
        score += 0.2
        indicators.append("unusual_vowel_ratio")
    
    return min(score, 5.0), indicators


# ============================================================================
# Sinkhole Detection
# ============================================================================

class SinkholeDetector:
    """Detect DNS sinkhole redirects to known malware sinkholes."""
    
    def __init__(self):
        """Initialize detector."""
        self.lock = threading.RLock()
        self.sinkhole_ips: Set[str] = set()
        self.sinkhole_asns: Set[str] = set()
        self.known_sinkholes: Dict[str, str] = {}
    
    def add_sinkhole_ip(self, ip: str, reason: str = "malware sinkhole") -> None:
        """Add known sinkhole IP."""
        with self.lock:
            self.sinkhole_ips.add(ip)
            self.known_sinkholes[ip] = reason
    
    def add_sinkhole_ips_bulk(self, ips: List[str]) -> None:
        """Add multiple sinkhole IPs."""
        with self.lock:
            self.sinkhole_ips.update(ips)
    
    def add_sinkhole_asn(self, asn: str) -> None:
        """Add ASN known for sinkholes."""
        with self.lock:
            self.sinkhole_asns.add(asn)
    
    def is_sinkhole(self, response_ip: str) -> Tuple[bool, str]:
        """
        Check if response IP is a known sinkhole.
        
        Returns (is_sinkhole, reason)
        """
        with self.lock:
            if response_ip in self.sinkhole_ips:
                reason = self.known_sinkholes.get(response_ip, "unknown sinkhole")
                return True, reason
        
        return False, ""
    
    def get_stats(self) -> Dict[str, int]:
        """Get detector statistics."""
        with self.lock:
            return {
                "total_sinkhole_ips": len(self.sinkhole_ips),
                "total_sinkhole_asns": len(self.sinkhole_asns),
            }


# ============================================================================
# DGA Detection
# ============================================================================

class DGADetector:
    """Detect Domain Generation Algorithm (DGA) domains."""
    
    def __init__(self, entropy_threshold: float = 4.0):
        """
        Initialize DGA detector.
        
        Args:
            entropy_threshold: Entropy threshold above which domain is suspect
        """
        self.entropy_threshold = entropy_threshold
        self.lock = threading.RLock()
        self.known_dga_domains: Set[str] = set()
        self.known_dga_patterns: List[str] = []
    
    def add_known_dga_domain(self, domain: str) -> None:
        """Add known DGA domain to database."""
        with self.lock:
            self.known_dga_domains.add(domain.lower())
    
    def add_dga_pattern(self, pattern: str) -> None:
        """Add regex pattern for DGA detection."""
        with self.lock:
            self.known_dga_patterns.append(pattern)
    
    def analyze(self, domain: str) -> Tuple[float, List[str]]:
        """
        Analyze domain for DGA characteristics.
        
        Returns (dga_score, indicators)
        Score: 0-5 scale, >4.0 is likely DGA
        """
        domain_lower = domain.lower()
        
        # Check known DGA domains
        if domain_lower in self.known_dga_domains:
            return 5.0, ["known_dga_domain"]
        
        # Check patterns
        with self.lock:
            for pattern in self.known_dga_patterns:
                try:
                    if re.match(pattern, domain):
                        return 4.5, ["matches_dga_pattern"]
                except re.error:
                    continue
        
        indicators = []
        score = 0.0
        
        # Entropy analysis
        entropy = calculate_domain_entropy(domain)
        if entropy > self.entropy_threshold:
            score += 1.0
            indicators.append(f"high_entropy_{entropy:.2f}")
        elif entropy > self.entropy_threshold - 0.5:
            score += 0.5
            indicators.append(f"moderate_entropy_{entropy:.2f}")
        
        # Pattern analysis
        pattern_score, pattern_indicators = analyze_domain_patterns(domain)
        score += pattern_score
        indicators.extend(pattern_indicators)
        
        # Length-based heuristics
        parts = domain.split(".")
        domain_name = parts[0] if parts else domain
        
        if len(domain_name) < 4:
            score -= 0.5  # Short names often legitimate
        
        # Remove obvious TLDs and check if name looks random
        tlds = {".com", ".org", ".net", ".edu", ".gov"}
        for tld in tlds:
            if domain.endswith(tld):
                # Analyze probability of this appearing in dictionary
                # Simplified: very long random strings are suspicious
                if len(domain_name) > 15 and entropy > 3.8:
                    score += 0.3
        
        return min(max(score, 0.0), 5.0), indicators


# ============================================================================
# DNS Tunneling Detection
# ============================================================================

class DNSTunnelingDetector:
    """Detect DNS tunneling and exfiltration attempts."""
    
    def __init__(self):
        """Initialize detector."""
        self.lock = threading.RLock()
        self.per_host_query_count = {}  # Track queries per host
        self.per_host_subdomains = {}   # Track unique subdomains per host
    
    def analyze_query_pattern(
        self,
        domain: str,
        source_ip: str,
        timestamp: float
    ) -> Tuple[float, List[str]]:
        """
        Analyze query for tunneling indicators.
        
        Returns (tunneling_score, indicators)
        Score: 0-1 scale
        """
        indicators = []
        score = 0.0
        
        # Check for base64/base32 encoding in subdomain
        parts = domain.split(".")
        
        for part in parts[:-1]:  # Check all but TLD
            # Base64 pattern (contains +, /, = or heavy vowel variation)
            if any(c in part for c in ["+", "/", "="]):
                indicators.append("base64_in_subdomain")
                score += 0.3
            
            # Base32 pattern (5-bit encoding, usually 32 chars max label)
            if len(part) > 8 and all(c in "abcdefghijklmnopqrstuvwxyz234567" for c in part):
                indicators.append("base32_pattern")
                score += 0.2
            
            # Hex encoding pattern
            if all(c in "0123456789abcdef" for c in part) and len(part) > 8:
                indicators.append("hex_pattern")
                score += 0.2
        
        # Check for subdomain enumeration (many subdomains, same domain)
        with self.lock:
            subdomain_key = ".".join(parts[-2:])  # domain.tld
            if source_ip not in self.per_host_subdomains:
                self.per_host_subdomains[source_ip] = {}
            
            if subdomain_key not in self.per_host_subdomains[source_ip]:
                self.per_host_subdomains[source_ip][subdomain_key] = set()
            
            self.per_host_subdomains[source_ip][subdomain_key].add(parts[0])
            
            unique_count = len(self.per_host_subdomains[source_ip][subdomain_key])
            
            # Many unique subdomains on same domain = tunneling
            if unique_count > 20:
                indicators.append(f"many_subdomains_{unique_count}")
                score += min(0.3, unique_count / 100)
        
        # Check domain label length (tunneling uses long labels for data)
        for part in parts:
            if len(part) > 30:
                indicators.append("very_long_label")
                score += 0.2
        
        return min(score, 1.0), indicators


# ============================================================================
# Policy Enforcement
# ============================================================================

class PolicyEnforcer:
    """Enforce DNS policy rules (blocklists, allowlists, RPZ)."""
    
    def __init__(self):
        """Initialize enforcer."""
        self.lock = threading.RLock()
        self.blocklist: Set[str] = set()
        self.allowlist: Set[str] = set()
        self.suspicious_tlds: Set[str] = set()
        self.rpz_rules: List[Tuple[str, str, str]] = []  # (pattern, action, reason)
    
    def add_blocklist_domain(self, domain: str) -> None:
        """Add domain to blocklist."""
        with self.lock:
            self.blocklist.add(domain.lower())
    
    def add_allowlist_domain(self, domain: str) -> None:
        """Add domain to allowlist."""
        with self.lock:
            self.allowlist.add(domain.lower())
    
    def add_suspicious_tld(self, tld: str) -> None:
        """Add suspicious TLD."""
        with self.lock:
            self.suspicious_tlds.add(tld.lower())
    
    def add_rpz_rule(self, pattern: str, action: str, reason: str) -> None:
        """
        Add RPZ-style rule.
        
        Args:
            pattern: Domain pattern (regex)
            action: Action type (block, warn, monitor)
            reason: Human-readable reason
        """
        with self.lock:
            self.rpz_rules.append((pattern, action, reason))
    
    def check_policy(self, domain: str) -> List[str]:
        """
        Check domain against policy rules.
        
        Returns list of policy violations
        """
        violations = []
        domain_lower = domain.lower()
        
        with self.lock:
            # Check blocklist
            if domain_lower in self.blocklist:
                violations.append("blocked_domain")
            
            # Check allowlist (override)
            if domain_lower in self.allowlist:
                return []
            
            # Check TLD
            parts = domain_lower.split(".")
            if len(parts) > 0:
                tld = parts[-1]
                if tld in self.suspicious_tlds:
                    violations.append(f"suspicious_tld_{tld}")
            
            # Check RPZ rules
            for pattern, action, reason in self.rpz_rules:
                try:
                    if re.match(pattern, domain_lower):
                        violations.append(f"{action}_{reason}")
                except re.error:
                    continue
        
        return violations
    
    def get_stats(self) -> Dict[str, int]:
        """Get policy statistics."""
        with self.lock:
            return {
                "blocklist_domains": len(self.blocklist),
                "allowlist_domains": len(self.allowlist),
                "suspicious_tlds": len(self.suspicious_tlds),
                "rpz_rules": len(self.rpz_rules),
            }


# ============================================================================
# DNS Detector Cache
# ============================================================================

class DNSDetectorCache:
    """Cache for DNS detection results."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        """Initialize cache."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
        self.cache: OrderedDict[str, Tuple[DNSAnalysisResult, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, domain: str) -> Optional[DNSAnalysisResult]:
        """Get cached result."""
        with self.lock:
            if domain not in self.cache:
                self.misses += 1
                return None
            
            result, timestamp = self.cache[domain]
            
            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[domain]
                self.misses += 1
                return None
            
            self.cache.move_to_end(domain)
            self.hits += 1
            return result
    
    def put(self, domain: str, result: DNSAnalysisResult) -> None:
        """Put result in cache."""
        with self.lock:
            if domain in self.cache:
                del self.cache[domain]
            
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[domain] = (result, time.time())
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


# ============================================================================
# Main DNS Detector
# ============================================================================

class DNSDetector:
    """Main DNS detection engine."""
    
    def __init__(
        self,
        sinkhole_ips: Optional[List[str]] = None,
        dga_entropy_threshold: float = 4.0,
        enable_cache: bool = True,
        cache_size: int = 10000
    ):
        """
        Initialize DNS detector.
        
        Args:
            sinkhole_ips: Known sinkhole IP list
            dga_entropy_threshold: Entropy threshold for DGA
            enable_cache: Whether to cache results
            cache_size: Cache size
        """
        self.sinkhole_detector = SinkholeDetector()
        self.dga_detector = DGADetector(entropy_threshold=dga_entropy_threshold)
        self.tunneling_detector = DNSTunnelingDetector()
        self.policy_enforcer = PolicyEnforcer()
        
        self.cache = DNSDetectorCache(max_size=cache_size) if enable_cache else None
        self.stats = DNSStats()
        self.lock = threading.RLock()
        
        # Load initial sinkhole IPs
        if sinkhole_ips:
            self.sinkhole_detector.add_sinkhole_ips_bulk(sinkhole_ips)
        
        logger.info(f"DNS detector initialized with {len(sinkhole_ips or [])} sinkhole IPs")
    
    def analyze_query(
        self,
        domain: str,
        response_ip: Optional[str] = None,
        query_type: str = "A",
        source_ip: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> DNSAnalysisResult:
        """
        Analyze DNS query.
        
        Args:
            domain: Domain name
            response_ip: Response IP (if any)
            query_type: Query type (A, AAAA, MX, etc.)
            source_ip: Source IP of query
            timestamp: Query timestamp
        
        Returns:
            DNSAnalysisResult with all analysis
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            self.stats.total_queries += 1
        
        # Check cache
        if self.cache:
            cached = self.cache.get(domain)
            if cached:
                return cached
        
        try:
            result = DNSAnalysisResult(
                domain=domain,
                query_type=query_type,
                response_ip=response_ip,
                timestamp=timestamp
            )
            
            # Sinkhole detection
            if response_ip:
                is_sink, reason = self.sinkhole_detector.is_sinkhole(response_ip)
                if is_sink:
                    result.is_sinkhole = True
                    result.sinkhole_reason = reason
                    with self.lock:
                        self.stats.sinkhole_detected += 1
            
            # DGA detection
            dga_score, dga_indicators = self.dga_detector.analyze(domain)
            result.dga_score = dga_score
            result.dga_indicators = dga_indicators
            if dga_score > 4.0:
                with self.lock:
                    self.stats.dga_detected += 1
            
            # Tunneling detection
            if source_ip:
                tunnel_score, tunnel_indicators = self.tunneling_detector.analyze_query_pattern(
                    domain, source_ip, timestamp
                )
                result.is_tunneling = tunnel_score > 0.5
                result.tunneling_score = tunnel_score
                result.tunneling_indicators = tunnel_indicators
                if result.is_tunneling:
                    with self.lock:
                        self.stats.tunneling_detected += 1
            
            # Policy enforcement
            violations = self.policy_enforcer.check_policy(domain)
            result.policy_violations = violations
            if violations:
                with self.lock:
                    self.stats.policy_violations += 1
            
            # Overall anomaly calculation
            result.anomaly_score = (
                (result.dga_score * 0.4) +
                (result.tunneling_score * 5.0 * 0.3) +
                (len(result.policy_violations) * 1.0) +
                (5.0 if result.is_sinkhole else 0.0) * 0.3
            )
            result.anomaly_score = min(result.anomaly_score, 5.0)
            
            if result.is_anomalous:
                with self.lock:
                    self.stats.anomalies += 1
            
            # Cache result
            if self.cache:
                self.cache.put(domain, result)
            
            return result
        
        except Exception as e:
            logger.error(f"DNS analysis error: {e}")
            with self.lock:
                self.stats.errors += 1
            raise
    
    def get_stats(self) -> DNSStats:
        """Get detection statistics."""
        with self.lock:
            stats = DNSStats(
                total_queries=self.stats.total_queries,
                sinkhole_detected=self.stats.sinkhole_detected,
                dga_detected=self.stats.dga_detected,
                tunneling_detected=self.stats.tunneling_detected,
                policy_violations=self.stats.policy_violations,
                anomalies=self.stats.anomalies,
                errors=self.stats.errors,
            )
            if self.cache:
                stats.cache_hits = self.cache.hits
                stats.cache_misses = self.cache.misses
            return stats


# ============================================================================
# Global Singleton
# ============================================================================

_dns_detector: Optional[DNSDetector] = None
_dns_lock = threading.Lock()


def get_dns_detector(
    sinkhole_ips: Optional[List[str]] = None,
    dga_entropy_threshold: float = 4.0
) -> DNSDetector:
    """Get or create global DNS detector instance."""
    global _dns_detector
    
    if _dns_detector is None:
        with _dns_lock:
            if _dns_detector is None:
                _dns_detector = DNSDetector(
                    sinkhole_ips=sinkhole_ips,
                    dga_entropy_threshold=dga_entropy_threshold
                )
    
    return _dns_detector


def init_dns_detector(
    sinkhole_ips: Optional[List[str]] = None,
    dga_entropy_threshold: float = 4.0
) -> DNSDetector:
    """Initialize and return global DNS detector."""
    return get_dns_detector(
        sinkhole_ips=sinkhole_ips,
        dga_entropy_threshold=dga_entropy_threshold
    )
