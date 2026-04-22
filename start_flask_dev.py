#!/usr/bin/env python
"""Development startup wrapper that enables auth bypass before Flask initialization."""

import os
import sys
import logging

# Set auth bypass BEFORE anything imports auth_service
os.environ['INIDS_ALLOW_UNAUTHENTICATED'] = '1'
print(f"DEBUG: INIDS_ALLOW_UNAUTHENTICATED = {os.environ.get('INIDS_ALLOW_UNAUTHENTICATED')}", file=sys.stderr)

# Enable DEBUG logging BEFORE importing Flask app
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s'
)

# Now import and run Flask app
from web_app.app import app

# Verify auth service initialized with bypass enabled
from src.auth_service import _auth_service
print(f"DEBUG: auth_service.allow_unauthenticated = {_auth_service.allow_unauthenticated}", file=sys.stderr)
print(f"DEBUG: auth_service.enabled = {_auth_service.enabled}", file=sys.stderr)

# CRITICAL: Ensure models are loaded at startup so ml_engine gets registered
print("DEBUG: Calling load_models() at startup to register ml_engine", file=sys.stderr)
from web_app.app import load_models, load_threat_intel, load_anomaly_baseline
load_models()

# Load threat intelligence indicators to enable TI engine
print("DEBUG: Calling load_threat_intel() at startup to enable threat_intel engine", file=sys.stderr)
load_threat_intel()

# Pre-fit anomaly engine with synthetic baseline to enable anomaly detection
print("DEBUG: Calling load_anomaly_baseline() at startup to enable anomaly engine", file=sys.stderr)
load_anomaly_baseline()

if __name__ == '__main__':
    # Run on 0.0.0.0:5000
    app.run(host='0.0.0.0', port=5000, debug=False)
