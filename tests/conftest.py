import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours

# Secure-by-default settings required at app import time.
os.environ.setdefault("SECRET_KEY", "unit-test-secret-key")
os.environ.setdefault("INIDS_ADMIN_API_KEY", "admin-token")
os.environ.setdefault("INIDS_SENSOR_API_KEY", "sensor-token")
os.environ.setdefault("INIDS_VIEWER_API_KEY", "viewer-token")
os.environ.setdefault("INIDS_ENABLE_IPS_SCHEDULER", "0")
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
