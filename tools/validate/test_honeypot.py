import requests

BASE_URL = "http://localhost:5000/api/detect"

def test_honeypot(ip):
    payload = {
        "features": {
            "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF", "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0, "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0, "count": 1, "srv_count": 1, "serror_rate": 0, "srv_serror_rate": 0, "rerror_rate": 0, "srv_rerror_rate": 0, "same_srv_rate": 0, "diff_srv_rate": 0, "srv_diff_host_rate": 0, "dst_host_count": 0, "dst_host_srv_count": 0, "dst_host_same_srv_rate": 0, "dst_host_diff_srv_rate": 0, "dst_host_same_src_port_rate": 0, "dst_host_srv_diff_host_rate": 0, "dst_host_serror_rate": 0, "dst_host_srv_serror_rate": 0, "dst_host_rerror_rate": 0, "dst_host_srv_rerror_rate": 0
        },
        "source": ip
    }
    
    response = requests.post(BASE_URL, json=payload)
    data = response.json()
    
    print(f"Source IP: {ip}")
    print(f"Verdict: {data.get('verdict')}")
    print(f"Severity: {data.get('severity')}")
    print(f"Confidence: {data.get('confidence')}%")
    print(f"Engines: {data.get('engine_count')}")
    
    for engine in data.get('engines', []):
        if engine['engine_id'] == 'honeypot':
            print(f"Honeypot Engine: {engine['verdict']} (Conf: {engine['confidence']}%) [Attack Type: {engine['attack_type']}]")

print("\n--- Testing Known Honeypot IP: 192.168.1.50 ---")
test_honeypot("192.168.1.50")

print("\n--- Testing Non-Honeypot IP: 192.168.1.100 ---")
test_honeypot("192.168.1.100")
