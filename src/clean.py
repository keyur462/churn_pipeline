from typing import Tuple

import pandas as pd

from config import (
    ALLOWED_CHURN_VALUES,
    ALLOWED_SUBSCRIPTION_TIERS,
    EXPECTED_COLUMNS,
    NUMERIC_RANGES,
)
from logger import get_logger

log = get_logger("clean")


def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Apply cleaning steps. Returns the cleaned DataFrame and a stats dict."""
    stats = {
        "rows_in":           len(df),
        "duplicates_removed": 0,
        "rows_dropped_invalid_numeric": 0,
        "rows_dropped_invalid_categorical": 0,
        "rows_out":          0,
    }

    # 1. Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # 2. Lowercase the categorical column so "Premium" and "premium" match
    if "subscription_tier" in df.columns:
        df["subscription_tier"] = df["subscription_tier"].str.lower()

    # 3. Parse dates (if columns exist) - errors='coerce' turns junk into NaT
    for date_col in ("signup_date", "last_login_date"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # 4. Coerce numeric columns - non-numeric junk becomes NaN
    for num_col in NUMERIC_RANGES:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # 5. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    stats["duplicates_removed"] = before - len(df)

    # 6. Drop rows with impossible numeric values (negative sessions, etc.)
    #    This catches obvious garbage; validation will decide if too many were
    #    dropped and the file is still trustworthy.
    before = len(df)
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col in df.columns:
            mask = df[col].isna() | df[col].between(lo, hi)
            df = df[mask]
    stats["rows_dropped_invalid_numeric"] = before - len(df)

    # 7. Drop rows with invalid categorical / label values
    #    Same idea as step 6 but for non-numeric columns.
    before = len(df)
    if "subscription_tier" in df.columns:
        mask = df["subscription_tier"].isna() | df["subscription_tier"].isin(ALLOWED_SUBSCRIPTION_TIERS)
        df = df[mask]
    if "churned" in df.columns:
        mask = df["churned"].isna() | df["churned"].isin(ALLOWED_CHURN_VALUES)
        df = df[mask]
    stats["rows_dropped_invalid_categorical"] = before - len(df)

    stats["rows_out"] = len(df)
    log.info(
        f"Cleaning done: {stats['rows_in']} -> {stats['rows_out']} rows "
        f"(dropped {stats['duplicates_removed']} dupes, "
        f"{stats['rows_dropped_invalid_numeric']} out-of-range, "
        f"{stats['rows_dropped_invalid_categorical']} bad-category)"
    )
    return df.reset_index(drop=True), stats
