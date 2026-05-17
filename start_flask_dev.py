#!/usr/bin/env python
"""Development startup wrapper."""

import os
import sys
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s'
)

from web_app.app import app

from web_app.app import load_models, load_threat_intel, load_anomaly_baseline
load_models()
load_threat_intel()
load_anomaly_baseline()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
