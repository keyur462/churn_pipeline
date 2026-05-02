from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from config import (
    MIN_ACCEPTABLE_ACCURACY,
    MODEL_RANDOM_STATE,
    MODEL_TEST_SIZE,
    MODELS_DIR,
)
from logger import get_logger

log = get_logger("train")

# Columns the model uses as features (everything except IDs, dates, and label).
FEATURE_COLUMNS = [
    "total_sessions",
    "avg_session_minutes",
    "subscription_tier",
    "monthly_spend",
    "support_tickets",
]
TARGET_COLUMN = "churned"


def train(df: pd.DataFrame, source_filename: str) -> dict:
    """Train a churn model and persist it. Returns a metrics dict."""
    log.info(f"Training on {len(df)} rows from {source_filename}")

    # Drop rows where the label is missing - can't train on those
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN]

    # One-hot encode the single categorical column
    X = pd.get_dummies(X, columns=["subscription_tier"], drop_first=False)

    # Median-impute any remaining numeric NaNs (validation already capped null %)
    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_TEST_SIZE,
        random_state=MODEL_RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    log.info(f"Test accuracy: {accuracy:.3f}")
    log.info("\n" + classification_report(y_test, preds, zero_division=0))

    if accuracy < MIN_ACCEPTABLE_ACCURACY:
        log.warning(
            f"Accuracy {accuracy:.3f} below acceptable threshold "
            f"{MIN_ACCEPTABLE_ACCURACY}. Model still saved for inspection."
        )

    # Persist the model alongside the feature schema (so we can re-encode at inference)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_path = MODELS_DIR / f"churn_model_{timestamp}.joblib"

    joblib.dump(
        {
            "model":           model,
            "feature_columns": list(X.columns),
            "trained_on":      source_filename,
            "trained_at":      timestamp,
            "test_accuracy":   accuracy,
        },
        model_path,
    )
    log.info(f"Model saved to {model_path}")

    return {
        "accuracy":    accuracy,
        "model_path":  str(model_path),
        "n_train":     len(X_train),
        "n_test":      len(X_test),
    }
