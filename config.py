from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

#: Bundled, human-readable sample of the processed dataset (used by the app
#: when the full processed dataset is not available locally).
SAMPLE_PREVIEW_PATH = DATA_DIR / "cicids2017_preview.csv"
SAMPLE_OVERVIEW_PATH = DATA_DIR / "cicids2017_overview.json"

#: Optional local copy of the full processed dataset (git-ignored).
FULL_DATASET_PATH = PROCESSED_DATA_DIR / "cicids2017.csv"

#: Static EDA chart images rendered on the Dataset Insights page.
DATASET_CHARTS_DIR = BASE_DIR / "CICIDS2017_Charts"

#: Default bundle written by ``train.py`` and loaded by the dashboard.
PRODUCTION_MODEL_NAME = "xgboost"
MODEL_BUNDLE_SUFFIX = ".bundle.joblib"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

TARGET_COLUMN = "Attack Type"

FEATURE_COLUMNS: tuple[str, ...] = (
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
    "Subflow Fwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Max",
    "Idle Min",
)

#: The seven traffic classes the classifier distinguishes.
CLASS_LABELS: tuple[str, ...] = (
    "Normal Traffic",
    "DoS",
    "DDoS",
    "Port Scanning",
    "Bots",
    "Brute Force",
    "Web Attacks",
)

#: Canonical label for benign traffic.
NORMAL_LABEL = "Normal Traffic"

#: Accepted spellings that normalise to :data:`NORMAL_LABEL`. Several public
#: CICIDS2017 mirrors use ``BENIGN`` instead of the project's label.
NORMAL_LABEL_ALIASES: frozenset[str] = frozenset(
    {
        "normal traffic",
        "normal",
        "benign",
        "benign traffic",
        "benign_traffic",
        "benign-traffic",
    }
)

#: Raw CICIDS2017 attack labels -> the project's merged class taxonomy.
#: The original project collapsed the 14 raw CICIDS2017 labels into 7
#: categories; this mapping reproduces that taxonomy from the public
#: MachineLearningCVE files.
LABEL_MAPPING: dict[str, str] = {
    "BENIGN": "Normal Traffic",
    "DDoS": "DDoS",
    "PortScan": "Port Scanning",
    "Bot": "Bots",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Web Attack – Brute Force": "Web Attacks",
    "Web Attack – XSS": "Web Attacks",
    "Web Attack – Sql Injection": "Web Attacks",
}

#: Raw labels that are intentionally excluded during preparation, with the
#: justification shown in reports and printed by ``prepare_data.py``.
EXCLUDED_RAW_LABELS: dict[str, str] = {
    "Heartbleed": (
        "11 flows in the whole dataset; no corresponding class in the "
        "project's 7-class taxonomy."
    ),
    "Infiltration": (
        "36 flows in the whole dataset; no corresponding class in the "
        "project's 7-class taxonomy."
    ),
}
DROPPED_FEATURES: dict[str, str] = {
    "Total Backward Packets": "schema",
    "Total Length of Bwd Packets": "schema",
    "Fwd PSH Flags": "sparse",
    "Bwd PSH Flags": "sparse",
    "Fwd URG Flags": "sparse",
    "Bwd URG Flags": "sparse",
    "SYN Flag Count": "sparse",
    "RST Flag Count": "sparse",
    "URG Flag Count": "sparse",
    "CWE Flag Count": "sparse",
    "ECE Flag Count": "sparse",
    "Down/Up Ratio": "correlated",
    "Avg Fwd Segment Size": "redundant",
    "Avg Bwd Segment Size": "redundant",
    "Fwd Header Length_duplicated_0": "redundant",
    "Fwd Avg Bytes/Bulk": "sparse",
    "Fwd Avg Packets/Bulk": "sparse",
    "Fwd Avg Bulk Rate": "sparse",
    "Bwd Avg Bytes/Bulk": "sparse",
    "Bwd Avg Packets/Bulk": "sparse",
    "Bwd Avg Bulk Rate": "sparse",
    "Subflow Fwd Packets": "redundant",
    "Subflow Fwd Bytes": "redundant",
    "Subflow Bwd Packets": "redundant",
    "Subflow Bwd Bytes": "redundant",
    "Active Std": "correlated",
    "Idle Std": "correlated",
}

# ---------------------------------------------------------------------------
# Severity / triage mapping used by the dashboard
# ---------------------------------------------------------------------------

SEVERITY_ORDER: tuple[str, ...] = ("Low", "Medium", "High", "Critical")

#: Class label (lower-case) -> triage severity.
CLASS_SEVERITY: dict[str, str] = {
    "normal traffic": "Low",
    "bots": "Medium",
    "port scanning": "Medium",
    "brute force": "High",
    "web attacks": "High",
    "dos": "Critical",
    "ddos": "Critical",
}

DEFAULT_SEVERITY = "Medium"

#: Severity -> analyst recommendation text.
SEVERITY_RECOMMENDATIONS: dict[str, str] = {
    "Low": (
        "This traffic pattern looks consistent with normal, benign activity. "
        "No action is required - continue routine monitoring."
    ),
    "Medium": (
        "This traffic shows characteristics of reconnaissance or automated "
        "activity (e.g. scanning or bot behavior). Review the source IP and "
        "consider rate-limiting or watch-listing it."
    ),
    "High": (
        "This traffic matches patterns associated with active exploitation "
        "attempts (e.g. brute force or web application attacks). Investigate "
        "the source promptly and consider blocking it at the firewall."
    ),
    "Critical": (
        "This traffic matches a denial-of-service pattern. Escalate "
        "immediately to the on-call security team and consider engaging "
        "upstream rate-limiting or scrubbing."
    ),
}

#: Network-health verdicts derived from the share of flagged flows.
HEALTH_THRESHOLDS: tuple[tuple[float, str, str], ...] = (
    (
        5.0,
        "Healthy",
        "No malicious traffic detected. Monitoring remains stable.",
    ),
    (
        20.0,
        "Low Risk",
        "A small number of suspicious flows were identified. Continue monitoring.",
    ),
    (
        40.0,
        "Medium Risk",
        "Suspicious traffic is elevated. Review the affected flows promptly.",
    ),
    (
        100.0,
        "High Risk",
        "A high number of malicious flows detected. Immediate investigation is recommended.",
    ),
)
