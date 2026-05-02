from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # CSVs "arrive" here
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned, validated data
FAILED_DIR = DATA_DIR / "failed"      # files that did not pass validation
MODELS_DIR = PROJECT_ROOT / "models"  # trained model artifacts
LOGS_DIR = PROJECT_ROOT / "logs"      # pipeline run logs

# Schema definition
# This is the single source of truth for what a "valid" CSV looks like.
# Validation rules are derived from this.
EXPECTED_COLUMNS = [
    "user_id",
    "signup_date",
    "last_login_date",
    "total_sessions",
    "avg_session_minutes",
    "subscription_tier",
    "monthly_spend",
    "support_tickets",
    "churned",
]

# Allowed values for categorical columns
ALLOWED_SUBSCRIPTION_TIERS = {"free", "basic", "premium", "enterprise"}
ALLOWED_CHURN_VALUES = {0, 1}

# Numeric column ranges (inclusive). Values outside these are invalid.
NUMERIC_RANGES = {
    "total_sessions":       (0, 10_000),
    "avg_session_minutes":  (0, 600),     # 10 hours is a generous upper bound
    "monthly_spend":        (0, 10_000),
    "support_tickets":      (0, 1_000),
}

# Data quality thresholds
# These are the "gates". If the data does not meet them, training is blocked.
MAX_NULL_FRACTION = 0.10        # no column may be more than 10% null
MAX_INVALID_ROW_FRACTION = 0.20  # if more than 20% of rows are bad, fail the file
MIN_ROW_COUNT = 50               # need at least this many rows to bother training

# Model settings
MODEL_RANDOM_STATE = 42
MODEL_TEST_SIZE = 0.2
MIN_ACCEPTABLE_ACCURACY = 0.60   # if model can't beat this, log a warning
