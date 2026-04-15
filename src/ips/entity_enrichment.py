"""
Entity Context Enrichment Engine
Enriches detected threats with contextual information from multiple sources:
- GeoIP data (location, ISP, ASN)
- Threat Intelligence lookups (IP reputation, known exploits)
- Historical attack patterns (prior incidents from same source)
- Network context (internal/external, VLAN, department)
"""

import logging
from dataclasses import dataclass, asdict
from typing import Optional, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class GeoIPContext:
    """Geographic and ISP information for an IP."""
    country: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    isp: Optional[str] = None
    asn: Optional[int] = None
    asn_org: Optional[str] = None
    is_vpn: bool = False
    is_proxy: bool = False
    is_datacenter: bool = False
    threat_level: str = "unknown"  # unknown, low, medium, high


@dataclass
class ThreatIntelContext:
    """Threat Intelligence enrichment for an IP."""
    ip_reputation_score: Optional[float] = None  # 0-100, higher = more malicious
    known_attacker: bool = False
    in_blacklist: bool = False
    in_honeypot_network: bool = False
    associated_malware: list[str] = None
    associated_campaigns: list[str] = None
    last_seen_attack_type: Optional[str] = None
    attack_count_30d: int = 0
    average_severity: Optional[float] = None


@dataclass
class HistoricalContext:
    """Historical attack patterns for an IP."""
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    total_incidents: int = 0
    incident_types: dict[str, int] = None  # {attack_type: count, ...}
    success_rate_percent: float = 0.0
    preferred_targets: list[str] = None
    attack_frequency_hours: Optional[float] = None  # Avg hours between attacks


@dataclass
class NetworkContext:
    """Internal network context for an IP."""
    is_internal: bool = False
    subnet: Optional[str] = None
    department: Optional[str] = None
    vlan_id: Optional[int] = None
    asset_name: Optional[str] = None
    asset_type: Optional[str] = None  # server, workstation, iot, printer, etc.
    criticality: str = "low"  # low, medium, high, critical
    last_known_user: Optional[str] = None


@dataclass
class EnrichedEntity:
    """Complete enriched context for a detected entity (IP address)."""
    ip_address: str
    timestamp: str
    geoip: GeoIPContext
    threat_intel: ThreatIntelContext
    historical: HistoricalContext
    network: NetworkContext
    enrichment_confidence: float = 0.5  # Weighted confidence score
    enrichment_errors: list[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert dataclasses to dicts
        data["geoip"] = asdict(self.geoip)
        data["threat_intel"] = asdict(self.threat_intel)
        if self.threat_intel.associated_malware is None:
            data["threat_intel"]["associated_malware"] = []
        if self.threat_intel.associated_campaigns is None:
            data["threat_intel"]["associated_campaigns"] = []
        data["historical"] = asdict(self.historical)
        if self.historical.incident_types is None:
            data["historical"]["incident_types"] = {}
        if self.historical.preferred_targets is None:
            data["historical"]["preferred_targets"] = []
        data["network"] = asdict(self.network)
        return data


class EntityEnrichmentEngine:
    """Multi-source enrichment engine for IP entities."""
    
    def __init__(self, ops_store=None, ti_manager=None, internal_cidrs: list[str] = None):
        """
        Initialize enrichment engine.
        
        Args:
            ops_store: OpsStore instance for historical lookups
            ti_manager: ThreatIntelManager instance for threat intel
            internal_cidrs: List of internal network CIDRs (e.g., ['192.168.0.0/16'])
        """
        self.ops_store = ops_store
        self.ti_manager = ti_manager
        self.internal_cidrs = internal_cidrs or [
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "127.0.0.0/8",
        ]
        self._geoip_cache = {}  # Simple cache: {ip: (context, timestamp)}
        self._cache_ttl_seconds = 3600  # 1 hour TTL
    
    def enrich(self, ip_address: str) -> EnrichedEntity:
        """
        Enrich an IP address with full context from all sources.
        
        Args:
            ip_address: IP address to enrich
            
        Returns:
            EnrichedEntity with complete context
        """
        timestamp = datetime.utcnow().isoformat()
        enrichment_errors = []
        
        # Parallel enrichment from multiple sources
        geoip_context = self._get_geoip_context(ip_address, enrichment_errors)
        threat_intel_context = self._get_threat_intel_context(ip_address, enrichment_errors)
        historical_context = self._get_historical_context(ip_address, enrichment_errors)
        network_context = self._get_network_context(ip_address, enrichment_errors)
        
        # Calculate composite confidence
        confidence = self._calculate_confidence(
            geoip_context, threat_intel_context, historical_context, network_context
        )
        
        entity = EnrichedEntity(
            ip_address=ip_address,
            timestamp=timestamp,
            geoip=geoip_context,
            threat_intel=threat_intel_context,
            historical=historical_context,
            network=network_context,
            enrichment_confidence=confidence,
            enrichment_errors=enrichment_errors,
        )
        
        logger.debug(f"Enriched entity {ip_address}: confidence={confidence:.2f}")
        return entity
    
    def _get_geoip_context(self, ip_address: str, errors: list) -> GeoIPContext:
        """Get GeoIP information (location, ISP, threat level)."""
        try:
            # Check cache
            if ip_address in self._geoip_cache:
                cached_context, cached_time = self._geoip_cache[ip_address]
                if (datetime.utcnow() - cached_time).total_seconds() < self._cache_ttl_seconds:
                    return cached_context
            
            # Dummy GeoIP lookup (in production, use MaxMind, IP2Location, etc.)
            context = GeoIPContext(
                country="US",
                city="Unknown",
                latitude=None,
                longitude=None,
                isp="Unknown ISP",
                asn=None,
                asn_org=None,
                is_vpn=self._is_vpn_ip(ip_address),
                is_proxy=self._is_proxy_ip(ip_address),
                is_datacenter=self._is_datacenter_ip(ip_address),
            )
            
            # Cache result
            self._geoip_cache[ip_address] = (context, datetime.utcnow())
            
            return context
        except Exception as e:
            logger.warning(f"GeoIP lookup failed for {ip_address}: {e}")
            errors.append(f"geoip_lookup_failed: {str(e)}")
            return GeoIPContext()
    
    def _get_threat_intel_context(self, ip_address: str, errors: list) -> ThreatIntelContext:
        """Get Threat Intelligence information from feeds and historical data."""
        try:
            context = ThreatIntelContext(
                associated_malware=[],
                associated_campaigns=[],
            )
            
            # Query threat intel manager if available
            if self.ti_manager:
                ti_data = self.ti_manager.query(ip_address)
                if ti_data:
                    context.ip_reputation_score = ti_data.get("reputation_score")
                    context.known_attacker = ti_data.get("known_attacker", False)
                    context.in_blacklist = ti_data.get("in_blacklist", False)
                    context.associated_malware = ti_data.get("malware", [])
                    context.associated_campaigns = ti_data.get("campaigns", [])
            
            # Query OpsStore for historical attack data
            if self.ops_store:
                try:
                    alerts = self.ops_store._fetchall(
                        f"SELECT attack_type FROM alerts WHERE source_ip = ? ORDER BY timestamp DESC LIMIT 50",
                        (ip_address,)
                    )
                    if alerts:
                        context.last_seen_attack_type = alerts[0].get("attack_type")
                        context.attack_count_30d = len([
                            a for a in alerts
                            if a.get("timestamp")
                        ])
                        severities = [a.get("severity", 0) for a in alerts]
                        if severities:
                            context.average_severity = sum(severities) / len(severities)
                except Exception:
                    pass
            
            return context
        except Exception as e:
            logger.warning(f"Threat Intel lookup failed for {ip_address}: {e}")
            errors.append(f"threat_intel_lookup_failed: {str(e)}")
            return ThreatIntelContext(associated_malware=[], associated_campaigns=[])
    
    def _get_historical_context(self, ip_address: str, errors: list) -> HistoricalContext:
        """Get historical attack patterns from ops store."""
        try:
            context = HistoricalContext(
                incident_types={},
                preferred_targets=[],
            )
            
            if not self.ops_store:
                return context
            
            # Get all incidents from this IP
            try:
                incidents = self.ops_store._fetchall(
                    f"SELECT source_ip, activity_count, timestamp FROM incidents WHERE source_ip = ? ORDER BY timestamp DESC LIMIT 100",
                    (ip_address,)
                )
                
                if incidents:
                    context.first_seen = incidents[-1].get("timestamp")  # Oldest
                    context.last_seen = incidents[0].get("timestamp")  # Newest
                    context.total_incidents = len(incidents)
                    
                    # Calculate attack frequency
                    if len(incidents) > 1:
                        first_ts = datetime.fromisoformat(incidents[-1].get("timestamp", datetime.utcnow().isoformat()))
                        last_ts = datetime.fromisoformat(incidents[0].get("timestamp", datetime.utcnow().isoformat()))
                        time_span = (last_ts - first_ts).total_seconds() / 3600  # Hours
                        if time_span > 0:
                            context.attack_frequency_hours = time_span / len(incidents)
                    
                    # Get incident types distribution
                    activity_counts = []
                    for inc in incidents:
                        activity_count = inc.get("activity_count", 0)
                        if activity_count > 0:
                            activity_counts.append(activity_count)
                    
                    if activity_counts:
                        successful = sum(1 for c in activity_counts if c > 0)
                        context.success_rate_percent = (successful / len(activity_counts)) * 100
            except Exception:
                pass
            
            return context
        except Exception as e:
            logger.warning(f"Historical lookup failed for {ip_address}: {e}")
            errors.append(f"historical_lookup_failed: {str(e)}")
            return HistoricalContext(incident_types={}, preferred_targets=[])
    
    def _get_network_context(self, ip_address: str, errors: list) -> NetworkContext:
        """Get internal network context (VLAN, department, asset info)."""
        try:
            context = NetworkContext()
            
            # Check if IP is internal
            context.is_internal = self._is_internal_ip(ip_address)
            
            if context.is_internal:
                # Look up in asset management (mock for now)
                context.asset_name = f"asset-{ip_address.split('.')[-1]}"
                context.asset_type = "workstation"
                context.criticality = "medium"
            
            return context
        except Exception as e:
            logger.warning(f"Network context lookup failed for {ip_address}: {e}")
            errors.append(f"network_lookup_failed: {str(e)}")
            return NetworkContext()
    
    def _calculate_confidence(
        self,
        geoip: GeoIPContext,
        threat_intel: ThreatIntelContext,
        historical: HistoricalContext,
        network: NetworkContext,
    ) -> float:
        """Calculate composite enrichment confidence score (0-1)."""
        scores = []
        weights = []
        
        # GeoIP confidence (0.3 weight)
        geoip_score = 1.0 if (geoip.country or geoip.isp) else 0.3
        scores.append(geoip_score)
        weights.append(0.3)
        
        # Threat Intel confidence (0.35 weight)
        ti_score = 0.5
        if threat_intel.ip_reputation_score is not None:
            ti_score = min(1.0, threat_intel.ip_reputation_score / 100.0)
        elif threat_intel.known_attacker or threat_intel.in_blacklist:
            ti_score = 0.9
        scores.append(ti_score)
        weights.append(0.35)
        
        # Historical confidence (0.2 weight)
        historical_score = min(1.0, historical.total_incidents / 50.0) if historical.total_incidents > 0 else 0.2
        scores.append(historical_score)
        weights.append(0.2)
        
        # Network confidence (0.15 weight)
        network_score = 0.7 if network.is_internal else 0.5
        scores.append(network_score)
        weights.append(0.15)
        
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight if total_weight > 0 else 0.5
        
        return min(1.0, max(0.0, weighted_score))
    
    def _is_internal_ip(self, ip_address: str) -> bool:
        """Check if IP is in internal CIDR ranges."""
        try:
            from ipaddress import ip_address as parse_ip, ip_network
            ip = parse_ip(ip_address)
            for cidr in self.internal_cidrs:
                if ip in ip_network(cidr, strict=False):
                    return True
        except Exception:
            pass
        return False
    
    def _is_vpn_ip(self, ip_address: str) -> bool:
        """Heuristic: Check if IP appears to be from a VPN provider."""
        # In production, query known VPN IP ranges
        return False
    
    def _is_proxy_ip(self, ip_address: str) -> bool:
        """Heuristic: Check if IP appears to be a proxy."""
        # In production, query known proxy IP ranges
        return False
    
    def _is_datacenter_ip(self, ip_address: str) -> bool:
        """Heuristic: Check if IP appears to be from a datacenter."""
        # In production, query datacenter IP ranges (AWS, Azure, GCP, etc.)
        return False
    
    def get_threat_level(self, entity: EnrichedEntity) -> str:
        """
        Calculate overall threat level from enriched entity.
        
        Returns: "low", "medium", "high", or "critical"
        """
        threat_score = 0.0
        
        # GeoIP risk
        if entity.geoip.threat_level == "high":
            threat_score += 0.2
        if entity.geoip.is_vpn or entity.geoip.is_proxy:
            threat_score += 0.15
        
        # Threat Intel risk
        if entity.threat_intel.known_attacker:
            threat_score += 0.3
        if entity.threat_intel.ip_reputation_score and entity.threat_intel.ip_reputation_score > 70:
            threat_score += 0.2
        if entity.threat_intel.in_blacklist:
            threat_score += 0.25
        
        # Historical risk
        if entity.historical.total_incidents > 10:
            threat_score += 0.2
        if entity.historical.success_rate_percent > 50:
            threat_score += 0.15
        
        # Determine level
        if threat_score >= 0.7:
            return "critical"
        elif threat_score >= 0.5:
            return "high"
        elif threat_score >= 0.3:
            return "medium"
        else:
            return "low"
