from __future__ import annotations

import argparse
import json
import os



DEFAULT_BASE_URL = os.getenv("INIDS_BASE_URL", "http://127.0.0.1:5000")


def run_demo(base_url: str, session=None) -> dict:
    if session is None:
        import requests
        session = requests.Session()

    health = session.get(f"{base_url}/api/health", timeout=10)
    health.raise_for_status()

    ingest = session.post(
        f"{base_url}/api/ingest",
        json={
            "source": "demo_replay",
            "rows": [
                {"duration": 1, "src_bytes": 200, "dst_bytes": 20, "count": 3, "srv_count": 2, "serror_rate": 0.0, "same_srv_rate": 0.8},
                {"duration": 2, "src_bytes": 5000, "dst_bytes": 10, "count": 20, "srv_count": 15, "serror_rate": 0.9, "same_srv_rate": 0.1},
            ],
        },
        timeout=10,
    )
    ingest.raise_for_status()

    process = session.post(f"{base_url}/api/ingest/process", json={"max_items": 20}, timeout=15)
    process.raise_for_status()

    alerts = session.get(f"{base_url}/api/alerts?limit=5", timeout=10)

    return {
        "health": health.json(),
        "ingest": ingest.json(),
        "processed": process.json(),
        "alerts_status": alerts.status_code,
        "alerts": alerts.json() if alerts.status_code == 200 else {"error": alerts.text},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run INIDS end-to-end demo flow")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    output = run_demo(args.base_url)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
