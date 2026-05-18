import requests

BASE_URL = "http://localhost:5000/api/detect"

def test_honeypot(src_ip, dst_ip=None, dst_port=None):
    payload = {
        "features": {
            "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF", "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0, "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0, "count": 1, "srv_count": 1, "serror_rate": 0, "srv_serror_rate": 0, "rerror_rate": 0, "srv_rerror_rate": 0, "same_srv_rate": 0, "diff_srv_rate": 0, "srv_diff_host_rate": 0, "dst_host_count": 0, "dst_host_srv_count": 0, "dst_host_same_srv_rate": 0, "dst_host_diff_srv_rate": 0, "dst_host_same_src_port_rate": 0, "dst_host_srv_diff_host_rate": 0, "dst_host_serror_rate": 0, "dst_host_srv_serror_rate": 0, "dst_host_rerror_rate": 0, "dst_host_srv_rerror_rate": 0
        },
        "source": src_ip
    }
    
    if dst_ip:
        payload["features"]["dst_ip"] = dst_ip
    if dst_port:
        payload["features"]["dst_port"] = dst_port
    
    response = requests.post(BASE_URL, json=payload)
    data = response.json()
    
    print(f"Testing - Source: {src_ip}, Dest IP: {dst_ip}, Dest Port: {dst_port}")
    print(f"Overall Verdict: {data.get('verdict')} (Severity: {data.get('severity')})")
    
    for engine in data.get('engines', []):
        if engine['engine_id'] == 'honeypot':
            print(f"  Honeypot Engine: {engine['verdict']} (Conf: {engine['confidence']}%) [Type: {engine['attack_type']}]")

print("\n--- Testing Destination Honeypot IP: 192.168.1.50 ---")
test_honeypot("192.168.1.100", dst_ip="192.168.1.50")

print("\n--- Testing Source Honeypot IP: 192.168.1.50 ---")
test_honeypot("192.168.1.50", dst_ip="192.168.1.100")

print("\n--- Testing Honeypot Port: 22 ---")
test_honeypot("192.168.1.100", dst_ip="192.168.1.200", dst_port=22)
