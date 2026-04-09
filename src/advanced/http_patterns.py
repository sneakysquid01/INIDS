"""
Phase F Part 4: HTTP Pattern Detection and Analysis

Detects HTTP attacks, anomalies, and malicious patterns:
- Signature-based malware detection
- Anomaly detection (unusual headers, encoding)
- Bot/scanner identification
- SQL injection and XSS pattern matching
- Payload encoding detection

Thread-safe with global singleton pattern.
"""

import re
import hashlib
import base64
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Set
from logging import getLogger
from collections import OrderedDict


logger = getLogger(__name__)

# HTTP Signature Patterns
HTTP_SIGNATURES = {
    "sql_injection": {
        "patterns": [
            rb"(\bUNION\b.*\bSELECT\b)",
            rb"(\bSELECT\b.*\bFROM\b.*\bWHERE\b)",
            rb"(\bDROP\b.*\bTABLE\b)",
            rb"(\bINSERT\b.*\bINTO\b)",
            rb"(\bDELETE\b.*\bFROM\b)",
            rb"(\bUPDATE\b.*\bSET\b)",
            rb"('.*'.*OR.*'1'='1')",
            rb"(;.*--|\*\s*--)",
        ],
        "severity": "critical",
    },
    "xss": {
        "patterns": [
            rb"(<script.*?>.*?</script>)",
            rb"(javascript:)",
            rb"(onerror=|onload=|onclick=)",
            rb"(<iframe.*?>)",
            rb"(<img.*?onerror=)",
        ],
        "severity": "critical",
    },
    "path_traversal": {
        "patterns": [
            rb"(\.\./|\.\.\\\)",
            rb"(%2e%2e/|%2e%2e\\)",
            rb"(\.\.%2f|\.\.%5c)",
        ],
        "severity": "high",
    },
    "command_injection": {
        "patterns": [
            rb"([;&|`].*(?:cat|ls|rm|bash|sh|cmd))",
            rb"(\$?\(.*(?:cat|ls|bash)\))",
        ],
        "severity": "high",
    },
}

# Scanner/Bot Detection Patterns
SCANNER_SIGNATURES = {
    "nmap": {
        "patterns": [
            r"(?i)nmap",
            r"(?i)masscan",
            r"(?i)nikto",
        ],
        "ua_patterns": [
            r"(?i)nmap",
            r"(?i)masscan",
        ],
    },
    "sqlmap": {
        "patterns": [
            r"(?i)sqlmap",
            r"(?i)union.*select",
        ],
        "ua_patterns": [
            r"(?i)sqlmap",
        ],
    },
    "metasploit": {
        "patterns": [
            r"(?i)metasploit",
            r"(?i)meterpreter",
        ],
        "ua_patterns": [
            r"(?i)metasploit",
        ],
    },
    "nuclei": {
        "patterns": [
            r"(?i)nuclei",
        ],
        "ua_patterns": [
            r"(?i)nuclei",
        ],
    },
    "web_crawler": {
        "patterns": [
            r"(?i)robot",
            r"(?i)crawler",
            r"(?i)spider",
            r"(?i)bot",
        ],
        "ua_patterns": [
            r"(?i)googlebotbot",
            r"(?i)bingbot",
            r"(?i)facebook",
            r"(?i)twitter",
        ],
    },
}

# Anomalous Headers
ANOMALOUS_HEADERS = {
    "Content-Length": [-1, 0],  # Anomalous lengths
    "Transfer-Encoding": [
        "chunked-chunked",
        "chunked; chunked",
    ],
    "Host": None,  # Will check for mismatch
    "User-Agent": ["", "None", "-"],
}

# Encoding patterns
ENCODING_INDICATORS = {
    "base64": rb"^[A-Za-z0-9+/]*={0,2}$",
    "hex": rb"^[0-9a-fA-F]*$",
    "url_encoded": rb"(%[0-9a-fA-F]{2})+",
    "unicode_encoded": rb"(\\u[0-9a-fA-F]{4})+",
}


@dataclass
class HTTPSignatureMatch:
    """Result of signature match."""
    signature_name: str
    pattern_type: str
    severity: str
    matched_pattern: str
    position: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


@dataclass
class HTTPAnomalyResult:
    """Result of anomaly detection."""
    anomaly_type: str
    description: str
    severity: str
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


@dataclass
class HTTPAnalysisResult:
    """Complete HTTP analysis result."""
    signatures_found: List[HTTPSignatureMatch] = field(default_factory=list)
    anomalies_detected: List[HTTPAnomalyResult] = field(default_factory=list)
    bot_detected: bool = False
    bot_type: Optional[str] = None
    bot_confidence: float = 0.0
    encoding_detected: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0-1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "signatures_found": [s.to_dict() for s in self.signatures_found],
            "anomalies_detected": [a.to_dict() for a in self.anomalies_detected],
            "bot_detected": self.bot_detected,
            "bot_type": self.bot_type,
            "bot_confidence": self.bot_confidence,
            "encoding_detected": self.encoding_detected,
            "risk_score": self.risk_score,
            "timestamp": self.timestamp,
        }


@dataclass
class HTTPStats:
    """HTTP analysis statistics."""
    total_analyzed: int = 0
    signatures_found: int = 0
    anomalies_detected: int = 0
    bots_detected: int = 0
    high_risk: int = 0
    critical_risk: int = 0
    last_analysis_time: float = 0.0


class HTTPPatternCache:
    """LRU cache with TTL for HTTP pattern analysis results."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[HTTPAnalysisResult]:
        with self.lock:
            if key not in self.cache:
                return None
            result, timestamp = self.cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                return None
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return result

    def put(self, key: str, result: HTTPAnalysisResult):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            self.cache[key] = (result, time.time())
            if len(self.cache) > self.max_size:
                # Remove oldest (least recently used)
                self.cache.popitem(last=False)

    def clear(self):
        with self.lock:
            self.cache.clear()


class HTTPSignatureDetector:
    """Detects HTTP signatures (malware, attacks)."""

    def __init__(self):
        self.signatures = HTTP_SIGNATURES
        self.lock = threading.RLock()

    def find_signatures(self, body: bytes) -> List[HTTPSignatureMatch]:
        """Find all signature matches in HTTP body."""
        matches = []

        with self.lock:
            for sig_name, sig_info in self.signatures.items():
                patterns = sig_info.get("patterns", [])
                severity = sig_info.get("severity", "medium")

                for pattern in patterns:
                    regex = re.compile(pattern, re.IGNORECASE)
                    for match in regex.finditer(body):
                        matches.append(
                            HTTPSignatureMatch(
                                signature_name=sig_name,
                                pattern_type="regex",
                                severity=severity,
                                matched_pattern=match.group(0).decode(
                                    "utf-8", errors="replace"
                                ),
                                position=match.start(),
                            )
                        )

        return matches


class BotDetector:
    """Detects bot/scanner traffic."""

    def __init__(self):
        self.scanners = SCANNER_SIGNATURES
        self.lock = threading.RLock()

    def detect_bot(
        self, user_agent: str = "", headers: Dict[str, str] = None, url: str = ""
    ) -> Tuple[bool, Optional[str], float]:
        """Detect bot/scanner. Returns (detected, bot_type, confidence)."""
        bot_type = None
        max_confidence = 0.0

        with self.lock:
            for scanner_name, scanner_info in self.scanners.items():
                confidence = 0.0

                # Check user-agent
                ua_patterns = scanner_info.get("ua_patterns", [])
                for pattern in ua_patterns:
                    if re.search(pattern, user_agent):
                        confidence = max(confidence, 0.8)
                        break

                # Check headers
                if headers:
                    for key, value in headers.items():
                        if isinstance(value, str):
                            for pattern in scanner_info.get("patterns", []):
                                if re.search(pattern, value):
                                    confidence = max(confidence, 0.6)

                # Check URL
                for pattern in scanner_info.get("patterns", []):
                    if re.search(pattern, url):
                        confidence = max(confidence, 0.5)

                if confidence > max_confidence:
                    max_confidence = confidence
                    bot_type = scanner_name

        return (max_confidence > 0.5, bot_type, max_confidence)


class HTTPAnomalyDetector:
    """Detects HTTP anomalies."""

    def __init__(self):
        self.lock = threading.RLock()

    def detect_anomalies(
        self,
        method: str = "",
        headers: Dict[str, str] = None,
        body: bytes = b"",
        url: str = "",
    ) -> List[HTTPAnomalyResult]:
        """Detect HTTP anomalies."""
        anomalies = []

        if headers is None:
            headers = {}

        with self.lock:
            # Check for missing User-Agent
            ua = headers.get("User-Agent", "")
            if not ua or ua in ["", "None", "-"]:
                anomalies.append(
                    HTTPAnomalyResult(
                        anomaly_type="missing_user_agent",
                        description="HTTP request missing User-Agent header",
                        severity="medium",
                        confidence=0.9,
                    )
                )

            # Check for excessive header count
            if len(headers) > 50:
                anomalies.append(
                    HTTPAnomalyResult(
                        anomaly_type="excessive_headers",
                        description=f"HTTP request has {len(headers)} headers (normal: <30)",
                        severity="low",
                        confidence=0.7,
                    )
                )

            # Check for suspiciously large body
            if len(body) > 100 * 1024 * 1024:  # >100MB
                anomalies.append(
                    HTTPAnomalyResult(
                        anomaly_type="large_body",
                        description=f"HTTP body size: {len(body)} bytes (abnormally large)",
                        severity="medium",
                        confidence=0.8,
                    )
                )

            # Check for unusual methods
            if method not in ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"]:
                anomalies.append(
                    HTTPAnomalyResult(
                        anomaly_type="unusual_method",
                        description=f"Unusual HTTP method: {method}",
                        severity="low",
                        confidence=0.6,
                    )
                )

            # Check for encoded payloads in suspicious locations
            if "?" in url:
                query = url.split("?", 1)[1]
                if self._is_heavily_encoded(query):
                    anomalies.append(
                        HTTPAnomalyResult(
                            anomaly_type="encoded_payload",
                            description="Query string is heavily encoded (possible obfuscation)",
                            severity="medium",
                            confidence=0.7,
                        )
                    )

            # Check Transfer-Encoding header
            te = headers.get("Transfer-Encoding", "")
            if "chunked" in te and te.count("chunked") > 1:
                anomalies.append(
                    HTTPAnomalyResult(
                        anomaly_type="double_encoding",
                        description="Suspicious Transfer-Encoding: double-chunked (HTTP smuggling)",
                        severity="critical",
                        confidence=0.95,
                    )
                )

        return anomalies

    def _is_heavily_encoded(self, text: str) -> bool:
        """Check if text is heavily URL or base64 encoded."""
        if not text:
            return False

        # Count encoded sequences
        encoded_count = 0
        if "%" in text:
            encoded_count = text.count("%")
        if "+" in text or text.count("=") >= 2:
            # Likely base64
            encoded_count = max(encoded_count, len(text) // 3)

        return encoded_count > len(text) * 0.3  # >30% encoded


class EncodingDetector:
    """Detects encoding in HTTP payloads."""

    def __init__(self):
        self.lock = threading.RLock()

    def detect_encodings(self, data: str) -> List[str]:
        """Detect encoding patterns in data."""
        encodings = []

        with self.lock:
            # Check base64
            if re.match(ENCODING_INDICATORS["base64"], data):
                if len(data) % 4 == 0:
                    encodings.append("base64")

            # Check hex
            if re.match(ENCODING_INDICATORS["hex"], data):
                if len(data) % 2 == 0 and len(data) >= 8:
                    encodings.append("hex")

            # Check URL encoding
            if re.search(ENCODING_INDICATORS["url_encoded"], data):
                encodings.append("url_encoded")

            # Check Unicode
            if re.search(ENCODING_INDICATORS["unicode_encoded"], data):
                encodings.append("unicode_encoded")

        return encodings


class HTTPPatternAnalyzer:
    """Main HTTP pattern analysis engine."""

    def __init__(
        self,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        min_risk_threshold: float = 0.3,
    ):
        self.cache = HTTPPatternCache(cache_size, cache_ttl)
        self.sig_detector = HTTPSignatureDetector()
        self.bot_detector = BotDetector()
        self.anomaly_detector = HTTPAnomalyDetector()
        self.encoding_detector = EncodingDetector()
        self.min_risk_threshold = min_risk_threshold
        self.stats = HTTPStats()
        self.lock = threading.RLock()

    def analyze(
        self,
        method: str = "",
        url: str = "",
        headers: Dict[str, str] = None,
        body: bytes = b"",
    ) -> HTTPAnalysisResult:
        """Analyze HTTP request/response."""

        if headers is None:
            headers = {}

        # Create cache key
        cache_key = hashlib.md5(
            f"{method}{url}{body[:100]}".encode()
        ).hexdigest()

        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        result = HTTPAnalysisResult()

        with self.lock:
            # Detect signatures
            result.signatures_found = self.sig_detector.find_signatures(body)

            # Detect anomalies
            result.anomalies_detected = self.anomaly_detector.detect_anomalies(
                method, headers, body, url
            )

            # Detect bot
            user_agent = headers.get("User-Agent", "")
            det, bot_type, confidence = self.bot_detector.detect_bot(
                user_agent, headers, url
            )
            result.bot_detected = det
            result.bot_type = bot_type
            result.bot_confidence = confidence

            # Detect encoding
            result.encoding_detected = self.encoding_detector.detect_encodings(
                url + body.decode("utf-8", errors="ignore")
            )

            # Calculate risk score
            result.risk_score = self._calculate_risk_score(result)

            # Update stats
            self.stats.total_analyzed += 1
            self.stats.signatures_found += len(result.signatures_found)
            self.stats.anomalies_detected += len(result.anomalies_detected)
            if result.bot_detected:
                self.stats.bots_detected += 1
            if result.risk_score >= 0.7:
                self.stats.high_risk += 1
            if result.risk_score >= 0.9:
                self.stats.critical_risk += 1
            self.stats.last_analysis_time = time.time()

        # Cache result
        self.cache.put(cache_key, result)

        return result

    def _calculate_risk_score(self, result: HTTPAnalysisResult) -> float:
        """Calculate risk score (0-1.0) based on findings."""
        score = 0.0

        # Signatures
        if result.signatures_found:
            for sig in result.signatures_found:
                if sig.severity == "critical":
                    score += 0.3
                elif sig.severity == "high":
                    score += 0.15
                elif sig.severity == "medium":
                    score += 0.05

        # Anomalies
        if result.anomalies_detected:
            for anomaly in result.anomalies_detected:
                if anomaly.severity == "critical":
                    score += 0.2
                elif anomaly.severity == "high":
                    score += 0.1
                elif anomaly.severity == "medium":
                    score += 0.05
                elif anomaly.severity == "low":
                    score += 0.02

        # Bot
        if result.bot_detected:
            score += 0.15

        # Encoding
        if len(result.encoding_detected) >= 2:
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def get_stats(self) -> HTTPStats:
        """Get analysis statistics."""
        with self.lock:
            return HTTPStats(
                total_analyzed=self.stats.total_analyzed,
                signatures_found=self.stats.signatures_found,
                anomalies_detected=self.stats.anomalies_detected,
                bots_detected=self.stats.bots_detected,
                high_risk=self.stats.high_risk,
                critical_risk=self.stats.critical_risk,
                last_analysis_time=self.stats.last_analysis_time,
            )

    def add_custom_signature(self, name: str, patterns: List[bytes], severity: str):
        """Add custom signature."""
        with self.lock:
            self.sig_detector.signatures[name] = {
                "patterns": patterns,
                "severity": severity,
            }


# Global singleton
_http_analyzer = None
_http_analyzer_lock = threading.RLock()


def get_http_analyzer() -> HTTPPatternAnalyzer:
    """Get global HTTP analyzer instance."""
    global _http_analyzer
    if _http_analyzer is None:
        with _http_analyzer_lock:
            if _http_analyzer is None:
                _http_analyzer = HTTPPatternAnalyzer()
    return _http_analyzer


def init_http_analyzer(
    cache_size: int = 10000, cache_ttl: int = 3600
) -> HTTPPatternAnalyzer:
    """Initialize HTTP analyzer with custom settings."""
    global _http_analyzer
    with _http_analyzer_lock:
        _http_analyzer = HTTPPatternAnalyzer(cache_size, cache_ttl)
    return _http_analyzer


def enrich_eve_event_with_http(
    eve_event: Dict, http_result: HTTPAnalysisResult
) -> Dict:
    """Enrich EVE JSON event with HTTP analysis."""
    if "http" not in eve_event:
        eve_event["http"] = {}

    eve_event["http"]["analysis"] = http_result.to_dict()

    # Add threat flags
    if http_result.risk_score >= 0.9:
        eve_event["http"]["threat_level"] = "critical"
    elif http_result.risk_score >= 0.7:
        eve_event["http"]["threat_level"] = "high"
    elif http_result.risk_score >= 0.5:
        eve_event["http"]["threat_level"] = "medium"
    elif http_result.risk_score >= 0.3:
        eve_event["http"]["threat_level"] = "low"
    else:
        eve_event["http"]["threat_level"] = "none"

    return eve_event
