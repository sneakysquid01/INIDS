#!/usr/bin/env python
"""
Comprehensive multi-engine detection test suite.
Tests normal + attack scenarios and validates EventBus chain.
"""

import requests
import json
import sys
from typing import Dict, Any, List

BASE_URL = "http://localhost:5000/api/detect"

# Base feature template (normal traffic)
BASE_FEATURES = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 0,
    "dst_bytes": 0,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 1,
    "srv_count": 1,
    "serror_rate": 0,
    "srv_serror_rate": 0,
    "rerror_rate": 0,
    "srv_rerror_rate": 0,
    "same_srv_rate": 0,
    "diff_srv_rate": 0,
    "srv_diff_host_rate": 0,
    "dst_host_count": 0,
    "dst_host_srv_count": 0,
    "dst_host_same_srv_rate": 0,
    "dst_host_diff_srv_rate": 0,
    "dst_host_same_src_port_rate": 0,
    "dst_host_srv_diff_host_rate": 0,
    "dst_host_serror_rate": 0,
    "dst_host_srv_serror_rate": 0,
    "dst_host_rerror_rate": 0,
    "dst_host_srv_rerror_rate": 0,
}

# Test scenarios
SCENARIOS = {
    "normal_http": {
        "name": "Normal HTTP traffic",
        "features": BASE_FEATURES.copy(),
        "attack_type": "normal",
    },
    "port_scan": {
        "name": "Port Scan (SYN scan with S flag)",
        "features": {
            **BASE_FEATURES,
            "protocol_type": "tcp",
            "flag": "S",  # SYN flag instead of SF
            "service": "http",
            "src_bytes": 0,  # Typical for scan
            "dst_bytes": 0,
            "count": 100,  # High count for reconnaissance
            "srv_count": 1,
            "serror_rate": 0.9,  # High SYN errors
            "srv_serror_rate": 0.9,
        },
        "attack_type": "port_scan",
    },
    "dos_attack": {
        "name": "DoS Attack (high packet rate)",
        "features": {
            **BASE_FEATURES,
            "protocol_type": "tcp",
            "flag": "SF",
            "service": "http",
            "src_bytes": 1000,
            "dst_bytes": 100,
            "count": 500,  # Abnormally high packet count
            "srv_count": 450,  # Most to same service
            "duration": 10,
            "serror_rate": 0.5,
            "same_srv_rate": 0.9,
            "diff_srv_rate": 0.01,
        },
        "attack_type": "dos",
    },
    "failed_login": {
        "name": "Failed Login Attempts",
        "features": {
            **BASE_FEATURES,
            "protocol_type": "tcp",
            "service": "login",  # SSH or telnet
            "flag": "SF",
            "num_failed_logins": 10,  # Multiple failed attempts
            "logged_in": 0,  # Not logged in
            "count": 20,
            "srv_count": 15,
        },
        "attack_type": "unauthorized_attempt",
    },
    "privilege_escalation": {
        "name": "Privilege Escalation (root_shell attempt)",
        "features": {
            **BASE_FEATURES,
            "protocol_type": "tcp",
            "service": "shell",
            "flag": "SF",
            "root_shell": 1,  # Root shell access detected
            "su_attempted": 1,  # Su command attempted
            "num_shells": 1,  # Shell spawned
        },
        "attack_type": "privilege_escalation",
    },
    "file_creation_attack": {
        "name": "Suspicious File Creation",
        "features": {
            **BASE_FEATURES,
            "protocol_type": "tcp",
            "service": "http",
            "num_file_creations": 100,  # Abnormal file creation
            "num_access_files": 50,  # File access
            "root_shell": 1,
        },
        "attack_type": "malware",
    },
    "high_entropy": {
        "name": "High Entropy Traffic (encrypted/tunneled)",
        "features": {
            **BASE_FEATURES,
            "src_bytes": 10000,
            "dst_bytes": 5000,
            "duration": 300,
            "service": "ssh",  # SSH with high throughput
        },
        "attack_type": "data_exfiltration",
    },
}

def test_scenario(name: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single scenario and return results."""
    payload = {
        "features": scenario["features"],
        "source": "192.168.1.100",
    }
    
    try:
        response = requests.post(
            BASE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        result = {
            "scenario": name,
            "description": scenario["name"],
            "status": response.status_code,
            "success": response.status_code == 200,
        }
        
        if response.status_code == 200:
            data = response.json()
            result["verdict"] = data.get("verdict")
            result["confidence"] = data.get("confidence")
            result["severity"] = data.get("severity")
            result["engine_count"] = data.get("engine_count", 0)
            result["engines"] = [
                {
                    "id": e.get("engine_id"),
                    "type": e.get("engine_type"),
                    "verdict": e.get("verdict"),
                    "confidence": e.get("confidence"),
                    "attack_type": e.get("attack_type"),
                }
                for e in data.get("engines", [])
            ]
            result["metadata"] = data.get("metadata", {})
        else:
            result["error"] = response.text
        
        return result
    
    except Exception as e:
        return {
            "scenario": name,
            "description": scenario["name"],
            "success": False,
            "error": str(e),
        }


def print_result(result: Dict[str, Any]) -> None:
    """Pretty-print a test result."""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {result['scenario']}")
    print(f"Description: {result['description']}")
    print(f"Status: {result.get('status', 'N/A')}")
    
    if result["success"]:
        print(f"\n  Verdict: {result['verdict']} (confidence: {result['confidence']}%)")
        print(f"  Severity: {result['severity']}")
        print(f"  Engines participating: {result['engine_count']}")
        
        if result["engines"]:
            print("\n  Engine Verdicts:")
            for engine in result["engines"]:
                print(
                    f"    • {engine['id']:20} ({engine['type']:10}): "
                    f"{engine['verdict']:10} (conf: {engine['confidence']:5.1f}%) "
                    f"[{engine['attack_type']}]"
                )
        else:
            print("  (No engines responded)")
        
        if result.get("metadata"):
            print(f"\n  Metadata: {json.dumps(result['metadata'], indent=4)}")
    else:
        print(f"  ERROR: {result['error']}")


def main():
    """Run all test scenarios."""
    print("="*70)
    print("MULTI-ENGINE DETECTION TEST SUITE")
    print("="*70)
    print(f"Base URL: {BASE_URL}")
    print(f"Total scenarios: {len(SCENARIOS)}")
    print("="*70)
    
    results: List[Dict[str, Any]] = []
    
    for scenario_key, scenario_data in SCENARIOS.items():
        print(f"\nTesting: {scenario_data['name']}...", end=" ", flush=True)
        result = test_scenario(scenario_key, scenario_data)
        results.append(result)
        
        status = "✓ OK" if result["success"] else "✗ FAILED"
        print(status)
    
    # Print detailed results
    print("\n\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    
    for result in results:
        print_result(result)
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r["success"])
    total_engines = sum(r.get("engine_count", 0) for r in results if r["success"])
    
    print(f"Successful requests: {success_count}/{len(results)}")
    print(f"Total engine verdicts: {total_engines}")
    
    # Engine participation analysis
    engine_participation = {}
    for result in results:
        if result["success"]:
            for engine in result.get("engines", []):
                eid = engine["id"]
                if eid not in engine_participation:
                    engine_participation[eid] = {"count": 0, "verdicts": {}}
                engine_participation[eid]["count"] += 1
                verdict = engine["verdict"]
                engine_participation[eid]["verdicts"][verdict] = \
                    engine_participation[eid]["verdicts"].get(verdict, 0) + 1
    
    print("\nEngine Participation:")
    for engine_id, data in sorted(engine_participation.items()):
        verdicts_str = ", ".join(f"{k}:{v}" for k, v in data["verdicts"].items())
        print(f"  • {engine_id:20} {data['count']:2}x  [{verdicts_str}]")
    
    missing_engines = {"ml_primary", "threat_intel", "honeypot", "anomaly"} - set(engine_participation.keys())
    if missing_engines:
        print(f"\nMissing/Not Responding: {', '.join(sorted(missing_engines))}")
    
    print("="*70)


if __name__ == "__main__":
    main()
