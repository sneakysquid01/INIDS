"""
Advanced Features Module - Phase F

Provides advanced detection and enrichment capabilities:
- GeoIP enrichment with VPN/proxy detection
- DNS attack detection (sinkhole, DGA, tunneling)
- TLS certificate validation
- HTTP signature patterns
- Machine learning anomaly detection
"""

from .geoip_enrichment import (
    GeoIPLookup,
    GeoIPData,
    GeoIPCache,
    GeoIPDatabase,
    GeoIPStats,
    RiskDetector,
    get_geoip_lookup,
    init_geoip,
    enrich_eve_event_with_geoip,
    ip_to_int,
    int_to_ip,
    is_private_ip,
    is_loopback_ip,
)

from .dns_detection import (
    DNSDetector,
    DNSAnalysisResult,
    DNSStats,
    SinkholeDetector,
    DGADetector,
    DNSTunnelingDetector,
    PolicyEnforcer,
    DNSDetectorCache,
    get_dns_detector,
    init_dns_detector,
    calculate_entropy,
    calculate_domain_entropy,
    analyze_domain_patterns,
)

from .tls_validation import (
    CertificateValidator,
    CertificateInfo,
    CertificateValidationResult,
    CertificateParser,
    TLSStats,
    get_tls_validator,
    init_tls_validator,
)

from .http_patterns import (
    HTTPPatternAnalyzer,
    HTTPSignatureDetector,
    BotDetector,
    HTTPAnomalyDetector,
    EncodingDetector,
    HTTPSignatureMatch,
    HTTPAnomalyResult,
    HTTPAnalysisResult,
    HTTPStats,
    HTTPPatternCache,
    get_http_analyzer,
    init_http_analyzer,
    enrich_eve_event_with_http,
)

from .ml_anomaly import (
    MLAnomalyDetector,
    StatisticalBaseline,
    BehavioralProfiler,
    EnsembleClassifier,
    CustomRuleEngine,
    HostProfile,
    AnomalyScore,
    MLDetectionResult,
    MLStats,
    get_ml_detector,
    init_ml_detector,
    enrich_eve_event_with_ml,
)

__all__ = [
    # GeoIP enrichment
    'GeoIPLookup',
    'GeoIPData',
    'GeoIPCache',
    'GeoIPDatabase',
    'GeoIPStats',
    'RiskDetector',
    'get_geoip_lookup',
    'init_geoip',
    'enrich_eve_event_with_geoip',
    'ip_to_int',
    'int_to_ip',
    'is_private_ip',
    'is_loopback_ip',
    
    # DNS detection
    'DNSDetector',
    'DNSAnalysisResult',
    'DNSStats',
    'SinkholeDetector',
    'DGADetector',
    'DNSTunnelingDetector',
    'PolicyEnforcer',
    'DNSDetectorCache',
    'get_dns_detector',
    'init_dns_detector',
    'calculate_entropy',
    'calculate_domain_entropy',
    'analyze_domain_patterns',
    
    # TLS validation
    'CertificateValidator',
    'CertificateInfo',
    'CertificateValidationResult',
    'CertificateParser',
    'TLSStats',
    'get_tls_validator',
    'init_tls_validator',
    
    # HTTP pattern detection
    'HTTPPatternAnalyzer',
    'HTTPSignatureDetector',
    'BotDetector',
    'HTTPAnomalyDetector',
    'EncodingDetector',
    'HTTPSignatureMatch',
    'HTTPAnomalyResult',
    'HTTPAnalysisResult',
    'HTTPStats',
    'HTTPPatternCache',
    'get_http_analyzer',
    'init_http_analyzer',
    'enrich_eve_event_with_http',
    
    # ML & anomaly detection
    'MLAnomalyDetector',
    'StatisticalBaseline',
    'BehavioralProfiler',
    'EnsembleClassifier',
    'CustomRuleEngine',
    'HostProfile',
    'AnomalyScore',
    'MLDetectionResult',
    'MLStats',
    'get_ml_detector',
    'init_ml_detector',
    'enrich_eve_event_with_ml',
]
