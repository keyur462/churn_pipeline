from dataclasses import dataclass, field
from typing import List

import pandas as pd

from config import (
    ALLOWED_CHURN_VALUES,
    ALLOWED_SUBSCRIPTION_TIERS,
    EXPECTED_COLUMNS,
    MAX_INVALID_ROW_FRACTION,
    MAX_NULL_FRACTION,
    MIN_ROW_COUNT,
)
from logger import get_logger

log = get_logger("validate")


@dataclass
class ValidationResult:
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)
        log.error(f"VALIDATION FAIL: {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning(f"VALIDATION WARN: {msg}")


def validate(df: pd.DataFrame, clean_stats: dict) -> ValidationResult:
    result = ValidationResult()

    # --- 1. Schema check -----------------------------------------------------
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        result.fail(f"Missing required columns: {sorted(missing)}")
        # If schema is broken, downstream checks will be noisy. Bail early.
        return result

    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        result.warn(f"Unexpected extra columns: {sorted(extra)}")

    # --- 2. Volume check -----------------------------------------------------
    if len(df) < MIN_ROW_COUNT:
        result.fail(f"Only {len(df)} rows after cleaning; need >= {MIN_ROW_COUNT}")

    # --- 3. Null fraction per column ----------------------------------------
    null_fractions = df.isna().mean()
    for col, frac in null_fractions.items():
        if frac > MAX_NULL_FRACTION:
            result.fail(
                f"Column '{col}' is {frac:.0%} null "
                f"(threshold: {MAX_NULL_FRACTION:.0%})"
            )

    # --- 4. Domain checks for categoricals ----------------------------------
    if "subscription_tier" in df.columns:
        bad_tiers = set(df["subscription_tier"].dropna().unique()) - ALLOWED_SUBSCRIPTION_TIERS
        if bad_tiers:
            result.fail(f"Unexpected subscription tiers: {sorted(bad_tiers)}")

    if "churned" in df.columns:
        bad_labels = set(df["churned"].dropna().unique()) - ALLOWED_CHURN_VALUES
        if bad_labels:
            result.fail(f"Invalid churn label values: {sorted(bad_labels)}")

    # --- 5. Drop rate during cleaning ---------------------------------------
    rows_in = clean_stats.get("rows_in", len(df))
    if rows_in > 0:
        drop_rate = 1 - (len(df) / rows_in)
        result.stats["drop_rate"] = drop_rate
        if drop_rate > MAX_INVALID_ROW_FRACTION:
            result.fail(
                f"Cleaning dropped {drop_rate:.0%} of rows "
                f"(threshold: {MAX_INVALID_ROW_FRACTION:.0%})"
            )
        elif drop_rate > 0.05:
            result.warn(f"Cleaning dropped {drop_rate:.0%} of rows")

    result.stats["row_count"] = len(df)
    result.stats["null_fractions"] = null_fractions.to_dict()

    if result.passed:
        log.info(f"Validation PASSED ({len(df)} rows, {len(result.warnings)} warnings)")
    else:
        log.error(f"Validation FAILED with {len(result.errors)} error(s)")

    return result
