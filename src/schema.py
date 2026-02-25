"""Shared data schema for NSL-KDD train/inference paths."""

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

LABEL_COLUMNS = ["label", "difficulty_level"]
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
FEATURE_COLUMNS = [col for col in COLUMNS if col not in LABEL_COLUMNS]
NUMERIC_FEATURES = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]

# Baseline values used for synthetic single-row inference (web form/live defaults).
DEFAULT_FEATURE_ROW = {
    "duration": 0.0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 0.0,
    "dst_bytes": 0.0,
    "land": 0.0,
    "wrong_fragment": 0.0,
    "urgent": 0.0,
    "hot": 0.0,
    "num_failed_logins": 0.0,
    "logged_in": 1.0,
    "num_compromised": 0.0,
    "root_shell": 0.0,
    "su_attempted": 0.0,
    "num_root": 0.0,
    "num_file_creations": 0.0,
    "num_shells": 0.0,
    "num_access_files": 0.0,
    "num_outbound_cmds": 0.0,
    "is_host_login": 0.0,
    "is_guest_login": 0.0,
    "count": 0.0,
    "srv_count": 0.0,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 0.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 0.0,
    "dst_host_srv_count": 0.0,
    "dst_host_same_srv_rate": 0.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.0,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0,
}
