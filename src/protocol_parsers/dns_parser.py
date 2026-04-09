"""
DNS Protocol Parser
Extracts DNS query/response details from UDP payload on port 53
Detects DNS tunneling, DGA, exfiltration patterns
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import struct
import re
import logging

logger = logging.getLogger(__name__)


class DNSRecordType(Enum):
    """DNS record types"""
    A = 1
    NS = 2
    CNAME = 5
    SOA = 6
    PTR = 12
    MX = 15
    TXT = 16
    AAAA = 28
    SRV = 33
    OTHER = 255


class DNSResponseCode(Enum):
    """DNS response codes"""
    NOERROR = 0    # Success
    FORMERR = 1    # Format error
    SERVFAIL = 2   # Server failure
    NXDOMAIN = 3   # Non-existent domain
    NOTIMP = 4     # Not implemented
    REFUSED = 5    # Query refused
    OTHER = 255


@dataclass
class DNSQuery:
    """Extracted DNS query details"""
    transaction_id: int                 # DNS transaction ID (2 bytes)
    domain: str                         # Queried domain
    query_type: str                     # A, AAAA, MX, CNAME, TXT, etc.
    query_class: str = "IN"             # Internet class (always IN for normal DNS)
    is_recursive: bool = False          # Recursion desired flag
    is_authoritative: bool = False      # Authoritative answer
    
    # Detection features
    domain_entropy: float = 0.0         # Entropy of domain name (high = DGA)
    is_suspicious: bool = False
    suspicious_indicators: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"DNSQuery({self.domain} {self.query_type})"


@dataclass
class DNSResponse:
    """Extracted DNS response details"""
    transaction_id: int                 # DNS transaction ID (must match query)
    response_code: str                  # NOERROR, NXDOMAIN, SERVFAIL, etc.
    query_domain: str                   # Original queried domain
    query_type: str
    
    # Response records
    answers: List[Dict] = field(default_factory=list)  # Answer RRs
    answer_ips: List[str] = field(default_factory=list)  # IPs from A/AAAA records
    answer_hostnames: List[str] = field(default_factory=list)  # From CNAME/MX/etc
    answer_txt_records: List[str] = field(default_factory=list)  # TXT record content
    
    is_error: bool = False              # Error response
    is_nxdomain: bool = False           # Domain doesn't exist
    is_refused: bool = False            # Query refused
    
    # Detection features
    is_suspicious: bool = False
    suspicious_indicators: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"DNSResponse({self.query_domain} {len(self.answer_ips)} IPs)"


class DNSParser:
    """Parse DNS protocol from UDP payload"""
    
    # DGA detection: high entropy domains (often contain many consonants)
    DGA_PATTERNS = [
        r'^[a-z0-9]{15,}$',  # Long random string
        r'[bcdfghjklmnpqrstvwxyz]{4,}',  # Many consonants in sequence
    ]
    
    # DNS tunneling detection patterns
    TUNNELING_PATTERNS = [
        r'[a-z0-9]{20,}\..*\.com$',  # Excessively long subdomain
        r'^([a-z0-9]{10,}\.)+[a-z]+$',  # Many dot-separated components
    ]
    
    # Suspicious TLDs/domains
    SUSPICIOUS_DOMAINS = {
        'test', 'example', 'local', 'arpa', 'localhost',
        'dga', 'botnet', 'c2', 'malware', 'cnc'
    }
    
    @staticmethod
    def parse_dns_query(payload: bytes, src_port: int = 0, dst_port: int = 0) -> Optional[DNSQuery]:
        """
        Parse DNS query from UDP payload
        
        Args:
            payload: UDP payload bytes (typically from port 53)
            src_port: Source port (for validation)
            dst_port: Destination port (should be 53 for queries)
        
        Returns:
            DNSQuery object or None if parsing fails
        """
        if not payload or len(payload) < 12:
            return None
        
        try:
            # Parse DNS header (12 bytes)
            trans_id, flags, qdcount, _, _, _ = struct.unpack('!HHHHHH', payload[:12])
            
            # Extract flags
            qr = (flags >> 15) & 1  # Query (0) or Response (1)
            if qr == 1:  # This is a response, not a query
                return None
            
            rd = (flags >> 8) & 1   # Recursion desired
            
            if qdcount < 1:
                return None
            
            # Parse question section
            offset = 12
            domain, new_offset = DNSParser._parse_domain_name(payload, offset)
            
            if not domain or new_offset + 4 > len(payload):
                return None
            
            offset = new_offset
            qtype, qclass = struct.unpack('!HH', payload[offset:offset+4])
            
            # Convert record type
            query_type = DNSParser._get_record_type_name(qtype)
            
            # Analyze domain for suspicion
            domain_entropy = DNSParser._calculate_entropy(domain)
            
            query = DNSQuery(
                transaction_id=trans_id,
                domain=domain,
                query_type=query_type,
                query_class='IN' if qclass == 1 else 'OTHER',
                is_recursive=bool(rd),
                domain_entropy=domain_entropy
            )
            
            # Check for suspicious patterns
            DNSParser._check_query_suspicious(query)
            
            return query
        
        except Exception as e:
            logger.debug(f"DNS query parse error: {e}")
            return None
    
    @staticmethod
    def parse_dns_response(payload: bytes) -> Optional[DNSResponse]:
        """
        Parse DNS response from UDP payload
        
        Args:
            payload: UDP payload bytes
        
        Returns:
            DNSResponse object or None if parsing fails
        """
        if not payload or len(payload) < 12:
            return None
        
        try:
            # Parse DNS header
            trans_id, flags, qdcount, ancount, nscount, arcount = struct.unpack('!HHHHHH', payload[:12])
            
            # Extract flags
            qr = (flags >> 15) & 1  # Must be 1 for response
            if qr == 0:  # This is a query, not a response
                return None
            
            rcode = flags & 0x0F    # Response code (bottom 4 bits)
            
            offset = 12
            
            # Parse question section
            query_domain = ""
            query_type = ""
            
            if qdcount > 0:
                query_domain, offset = DNSParser._parse_domain_name(payload, offset)
                if offset + 4 > len(payload):
                    return None
                qtype, _ = struct.unpack('!HH', payload[offset:offset+4])
                query_type = DNSParser._get_record_type_name(qtype)
                offset += 4
            
            # Parse answer section
            answers = []
            answer_ips = []
            answer_hostnames = []
            answer_txt_records = []
            
            for _ in range(ancount):
                domain, offset = DNSParser._parse_domain_name(payload, offset)
                if offset + 10 > len(payload):
                    break
                
                rrtype, rrclass, ttl, rdlen = struct.unpack('!HHIH', payload[offset:offset+10])
                offset += 10
                
                if offset + rdlen > len(payload):
                    break
                
                rdata = payload[offset:offset+rdlen]
                offset += rdlen
                
                rr_type_name = DNSParser._get_record_type_name(rrtype)
                
                # Parse specific record types
                if rrtype == 1:  # A record
                    if rdlen == 4:
                        ip = '.'.join(str(b) for b in rdata)
                        answer_ips.append(ip)
                        answers.append({'type': 'A', 'value': ip, 'domain': domain})
                
                elif rrtype == 28:  # AAAA record
                    if rdlen == 16:
                        ip = ':'.join(f'{b:02x}' for b in rdata)
                        answer_ips.append(ip)
                        answers.append({'type': 'AAAA', 'value': ip, 'domain': domain})
                
                elif rrtype == 5:  # CNAME record
                    cname, _ = DNSParser._parse_domain_name(rdata, 0)
                    if cname:
                        answer_hostnames.append(cname)
                        answers.append({'type': 'CNAME', 'value': cname, 'domain': domain})
                
                elif rrtype == 15:  # MX record
                    if rdlen >= 2:
                        mx_domain, _ = DNSParser._parse_domain_name(rdata, 2)
                        if mx_domain:
                            answer_hostnames.append(mx_domain)
                            answers.append({'type': 'MX', 'value': mx_domain, 'domain': domain})
                
                elif rrtype == 16:  # TXT record
                    txt_value = rdata.decode('utf-8', errors='ignore')
                    answer_txt_records.append(txt_value)
                    answers.append({'type': 'TXT', 'value': txt_value, 'domain': domain})
                
                else:
                    answers.append({'type': rr_type_name, 'value': rdata.hex(), 'domain': domain})
            
            # Determine response code
            response_code = DNSParser._get_response_code_name(rcode)
            
            response = DNSResponse(
                transaction_id=trans_id,
                response_code=response_code,
                query_domain=query_domain,
                query_type=query_type,
                answers=answers,
                answer_ips=answer_ips,
                answer_hostnames=answer_hostnames,
                answer_txt_records=answer_txt_records
            )
            
            # Set error flags
            response.is_error = rcode != 0
            response.is_nxdomain = rcode == 3
            response.is_refused = rcode == 5
            
            # Check for suspicious patterns
            DNSParser._check_response_suspicious(response)
            
            return response
        
        except Exception as e:
            logger.debug(f"DNS response parse error: {e}")
            return None
    
    @staticmethod
    def _parse_domain_name(data: bytes, offset: int) -> tuple[str, int]:
        """
        Parse DNS domain name from payload (handles compression pointers)
        
        Returns:
            Tuple of (domain_name, new_offset)
        """
        labels = []
        start_offset = offset
        
        while offset < len(data):
            length_byte = data[offset]
            offset += 1
            
            if length_byte == 0:
                # End of domain name
                break
            
            elif (length_byte & 0xC0) == 0xC0:
                # Pointer (compression)
                if offset >= len(data):
                    return '', start_offset
                
                pointer = ((length_byte & 0x3F) << 8) | data[offset]
                offset += 1
                
                # Recursively follow pointer
                ptr_domain, _ = DNSParser._parse_domain_name(data, pointer)
                if ptr_domain:
                    labels.append(ptr_domain)
                break
            
            else:
                # Regular label
                if offset + length_byte > len(data):
                    return '', start_offset
                
                label = data[offset:offset+length_byte].decode('utf-8', errors='ignore')
                labels.append(label)
                offset += length_byte
        
        domain = '.'.join(labels)
        return domain, offset
    
    @staticmethod
    def _calculate_entropy(domain: str) -> float:
        """Calculate Shannon entropy of domain name (indicator of DGA)"""
        if not domain:
            return 0.0
        
        # Remove TLD
        parts = domain.split('.')
        if len(parts) > 1:
            domain_part = parts[0]
        else:
            domain_part = domain
        
        # Calculate entropy
        char_counts = {}
        for char in domain_part.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0.0
        domain_len = len(domain_part)
        
        for count in char_counts.values():
            prob = count / domain_len
            entropy -= prob * (prob ** 0.5)  # Simplified entropy
        
        return entropy
    
    @staticmethod
    def _check_query_suspicious(query: DNSQuery):
        """Check for suspicious patterns in DNS query"""
        
        domain_lower = query.domain.lower()
        
        # Check for high entropy (DGA indicator)
        if query.domain_entropy > 3.5:
            query.suspicious_indicators.append("high_entropy_domain")
        
        # Check DGA patterns
        for pattern in DNSParser.DGA_PATTERNS:
            if re.search(pattern, domain_lower):
                query.suspicious_indicators.append("dga_pattern")
                break
        
        # Check tunneling patterns
        for pattern in DNSParser.TUNNELING_PATTERNS:
            if re.search(pattern, domain_lower):
                query.suspicious_indicators.append("dns_tunneling_pattern")
                break
        
        # Check very long domain (exfiltration)
        if len(query.domain) > 100:
            query.suspicious_indicators.append("excessive_domain_length")
        
        # Check suspicious TLD/keywords
        for suspicious in DNSParser.SUSPICIOUS_DOMAINS:
            if suspicious in domain_lower:
                query.suspicious_indicators.append(f"suspicious_keyword_{suspicious}")
        
        # Check for numeric IP-like domain
        if query.domain.replace('.', '').isdigit():
            query.suspicious_indicators.append("ip_like_domain")
        
        # Check for uncommon TLD
        tld = domain_lower.split('.')[-1]
        common_tlds = {'com', 'org', 'net', 'edu', 'gov', 'io', 'uk', 'de', 'fr'}
        if len(tld) > 3 and tld not in common_tlds:
            query.suspicious_indicators.append("uncommon_tld")
        
        query.is_suspicious = len(query.suspicious_indicators) > 0
    
    @staticmethod
    def _check_response_suspicious(response: DNSResponse):
        """Check for suspicious patterns in DNS response"""
        
        # NXDOMAIN response (domain doesn't exist)
        if response.is_nxdomain:
            response.suspicious_indicators.append("nxdomain_response")
        
        # Refused response
        if response.is_refused:
            response.suspicious_indicators.append("query_refused")
        
        # IPv6 (not inherently suspicious, but worth noting)
        if response.answer_ips:
            v6_ips = [ip for ip in response.answer_ips if ':' in ip]
            if len(v6_ips) > 0 and len(response.answer_ips) > 2:
                response.suspicious_indicators.append("multiple_ipv6_answers")
        
        # Large number of answers (possible poisoning)
        if len(response.answers) > 20:
            response.suspicious_indicators.append("excessive_answers")
        
        # Mismatched transaction ID would be detected at higher level
        
        response.is_suspicious = len(response.suspicious_indicators) > 0
    
    @staticmethod
    def _get_record_type_name(qtype: int) -> str:
        """Get DNS record type name"""
        try:
            return DNSRecordType(qtype).name
        except ValueError:
            return f"TYPE{qtype}"
    
    @staticmethod
    def _get_response_code_name(rcode: int) -> str:
        """Get DNS response code name"""
        try:
            return DNSResponseCode(rcode).name
        except ValueError:
            return f"UNKNOWN{rcode}"
    
    @staticmethod
    def extract_features(query: DNSQuery) -> Dict[str, any]:
        """Extract ML features from DNS query"""
        if not query:
            return {}
        
        return {
            "dns_domain_length": len(query.domain),
            "dns_domain_entropy": query.domain_entropy,
            "dns_query_type": query.query_type,
            "dns_is_recursive": query.is_recursive,
            "dns_is_suspicious": query.is_suspicious,
            "dns_suspicious_indicators_count": len(query.suspicious_indicators),
            "dns_label_count": len(query.domain.split('.')),
        }
    
    @staticmethod
    def extract_features_response(response: DNSResponse) -> Dict[str, any]:
        """Extract ML features from DNS response"""
        if not response:
            return {}
        
        return {
            "dns_response_code": response.response_code,
            "dns_answer_count": len(response.answers),
            "dns_answer_ips_count": len(response.answer_ips),
            "dns_is_nxdomain": response.is_nxdomain,
            "dns_is_refused": response.is_refused,
            "dns_is_suspicious": response.is_suspicious,
            "dns_suspicious_indicators_count": len(response.suspicious_indicators),
        }
