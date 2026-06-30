import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

CANDIDATES_JSONL_GZ = BASE_DIR / "candidates.jsonl.gz"
CANDIDATES_JSONL = BASE_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES_JSON = BASE_DIR / "sample_candidates.json"
JOB_DESCRIPTION_MD = BASE_DIR / "job_description.md"
REDROB_SIGNALS_MD = BASE_DIR / "redrob_signals_doc.md"

# Calibration weights JSON path
CALIBRATION_WEIGHTS_PATH = ARTIFACTS_DIR / "calibration_weights.json"

# Default scoring weights (will be calibrated/loaded from calibration_weights.json if it exists)
CORE_FIT_WEIGHT = 0.55
LOGISTICS_WEIGHT = 0.25
EDUCATION_WEIGHT = 0.20

# Gating Thresholds
MIN_YOE_FLOOR = 3.0  # JD says 5-9 but considers slightly below. Let's gate below 3.0
MAX_YOE_CEILING = 15.0 # Let's not strictly gate seniors unless they are pure managers
MAX_NOTICE_PERIOD_DAYS_GATE = 120  # Extreme notice period gate

# Signal Constants & Sentinels
OFFER_ACCEPTANCE_RATE_SENTINEL = -1.0
GITHUB_ACTIVITY_SCORE_SENTINEL = -1.0

# LLM Feature Extraction & Narrative settings
REDUCED_POOL_SIZE = 4000  # Number of candidates retrieved via FAISS to run LLM verification on
CROSS_ENCODER_TOP_K = 200 # Number of candidates to rerank via Cross-Encoder

# Embedded Signal Weightings (behavioral multipliers)
# 1. Recruiter response rate (0.0-1.0)
# 2. Last active date (days since reference date 2026-07-01)
# 3. Open to work flag
# 4. Profile completeness score

REFERENCE_DATE_STR = "2026-07-01"
