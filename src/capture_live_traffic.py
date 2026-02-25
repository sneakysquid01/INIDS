"""
Live Network Traffic Capture for INIDS
Captures real network packets and converts to analyzable format
"""
from scapy.all import sniff, IP, TCP, UDP, ICMP
import pandas as pd
import time
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURE_DIR = os.path.join(BASE_DIR, "captured_data")
os.makedirs(CAPTURE_DIR, exist_ok=True)

def extract_features(packet):
    """Extract NSL-KDD-like features from network packet"""
    try:
        features = {
            'duration': 0,
            'protocol_type': 'tcp',
            'service': 'http',
            'flag': 'SF',
            'src_bytes': 0,
            'dst_bytes': 0,
            'land': 0,
            'wrong_fragment': 0,
            'urgent': 0,
            'hot': 0,
            'num_failed_logins': 0,
            'logged_in': 1,
            'num_compromised': 0,
            'root_shell': 0,
            'su_attempted': 0,
            'num_root': 0,
            'num_file_creations': 0,
            'num_shells': 0,
            'num_access_files': 0,
            'num_outbound_cmds': 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': 1,
            'srv_count': 1,
            'serror_rate': 0.0,
            'srv_serror_rate': 0.0,
            'rerror_rate': 0.0,
            'srv_rerror_rate': 0.0,
            'same_srv_rate': 1.0,
            'diff_srv_rate': 0.0,
            'srv_diff_host_rate': 0.0,
            'dst_host_count': 1,
            'dst_host_srv_count': 1,
            'dst_host_same_srv_rate': 1.0,
            'dst_host_diff_srv_rate': 0.0,
            'dst_host_same_src_port_rate': 1.0,
            'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': 0.0,
            'dst_host_srv_serror_rate': 0.0,
            'dst_host_rerror_rate': 0.0,
            'dst_host_srv_rerror_rate': 0.0,
        }
        
        # Extract protocol type
        if TCP in packet:
            features['protocol_type'] = 'tcp'
            features['src_bytes'] = len(packet)
        elif UDP in packet:
            features['protocol_type'] = 'udp'
            features['src_bytes'] = len(packet)
        elif ICMP in packet:
            features['protocol_type'] = 'icmp'
            features['src_bytes'] = len(packet)
        
        # Extract service based on port (simplified)
        if TCP in packet:
            dport = packet[TCP].dport
            if dport == 80:
                features['service'] = 'http'
            elif dport == 443:
                features['service'] = 'https'
            elif dport == 21:
                features['service'] = 'ftp'
            elif dport == 22:
                features['service'] = 'ssh'
            elif dport == 25:
                features['service'] = 'smtp'
        
        return features
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

def capture_traffic(count=100, timeout=30):
    """
    Capture network packets from default interface
    Args:
        count: Number of packets to capture
        timeout: Maximum time to wait (seconds)
    Returns:
        Path to saved CSV file
    """
    logging.info(f"🔍 Starting packet capture: {count} packets, {timeout}s timeout")
    logging.info("📡 Listening on default network interface...")
    
    captured_packets = []
    
    def process_packet(packet):
        if IP in packet:
            features = extract_features(packet)
            if features:
                captured_packets.append(features)
                if len(captured_packets) % 10 == 0:
                    logging.info(f"   Captured {len(captured_packets)} packets...")
    
    try:
        # Capture packets
        sniff(count=count, timeout=timeout, prn=process_packet, store=False)
        
        if not captured_packets:
            logging.warning("⚠️  No packets captured! Check network activity.")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(captured_packets)
        
        # Save to CSV
        timestamp = int(time.time())
        filename = f"live_capture_{timestamp}.csv"
        filepath = os.path.join(CAPTURE_DIR, filename)
        df.to_csv(filepath, index=False)
        
        logging.info(f"✅ Successfully captured {len(captured_packets)} packets")
        logging.info(f"💾 Saved to: {filepath}")
        
        return filepath
        
    except PermissionError:
        logging.error("❌ Permission denied! Run as administrator/sudo for packet capture.")
        return None
    except Exception as e:
        logging.error(f"❌ Capture failed: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("INIDS - Live Network Traffic Capture")
    print("=" * 60)
    print("\n⚠️  Note: Packet capture requires elevated privileges")
    print("Windows: Run as Administrator")
    print("Linux/Mac: Run with sudo\n")
    
    capture_traffic(count=50, timeout=20)
