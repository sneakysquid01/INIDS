import os
import sys
from datetime import datetime, timezone

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ops_store import OpsStore
from src.ips.alert_filter import ThreeLayerAlertFilter
from src.ips.incident_aggregator import IncidentAggregator


@pytest.fixture
def temp_db(tmp_path):
    yield str(tmp_path / "test_ops.db")


@pytest.fixture
def mock_ops_store(temp_db):
    store = OpsStore(temp_db)
    IncidentAggregator(store)
    ThreeLayerAlertFilter(store)
    return store


@pytest.fixture
def honeypot_config():
    return {
        "honeypot_ips": ["10.10.10.10", "192.168.99.99"],
        "honeypot_ports": [22, 80, 443, 3389],
        "enabled": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
