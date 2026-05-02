from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import RAW_DIR

random.seed(42)
np.random.seed(42)

TIERS = ["free", "basic", "premium", "enterprise"]
TIER_WEIGHTS = [0.40, 0.30, 0.20, 0.10]


def _generate_clean_rows(n_rows: int, start_user_id: int = 1000) -> pd.DataFrame:
    """Build a DataFrame of well-formed user activity rows.

    Churn label is loosely tied to the features so the trained model can
    actually learn something - low sessions + low spend + many tickets
    raises the probability of churn.
    """
    rows = []
    today = datetime(2024, 1, 15)

    for i in range(n_rows):
        signup = today - timedelta(days=random.randint(30, 730))
        last_login = signup + timedelta(days=random.randint(1, (today - signup).days))
        tier = random.choices(TIERS, weights=TIER_WEIGHTS)[0]

        total_sessions = max(1, int(np.random.gamma(2.5, 15)))
        avg_session = round(np.random.gamma(2.0, 8), 2)

        spend_base = {"free": 0, "basic": 9.99, "premium": 29.99, "enterprise": 99.99}[tier]
        monthly_spend = round(spend_base + np.random.normal(0, 3), 2)
        monthly_spend = max(0, monthly_spend)

        support_tickets = np.random.poisson(1.2)

        # Loose causal signal for churn so the model learns a real pattern
        churn_score = (
            (total_sessions < 10) * 0.4
            + (monthly_spend < 5) * 0.3
            + (support_tickets > 3) * 0.3
            + np.random.uniform(-0.2, 0.2)
        )
        churned = 1 if churn_score > 0.5 else 0

        rows.append({
            "user_id":             f"U{start_user_id + i:06d}",
            "signup_date":         signup.strftime("%Y-%m-%d"),
            "last_login_date":     last_login.strftime("%Y-%m-%d"),
            "total_sessions":      total_sessions,
            "avg_session_minutes": avg_session,
            "subscription_tier":   tier,
            "monthly_spend":       monthly_spend,
            "support_tickets":     support_tickets,
            "churned":             churned,
        })

    return pd.DataFrame(rows)


def make_day1_clean() -> pd.DataFrame:
    """Day 1: textbook clean data. Should pass every check."""
    return _generate_clean_rows(n_rows=500, start_user_id=1000)


def make_day2_bad_schema() -> pd.DataFrame:
    """Day 2: missing the `churned` column entirely.

    This simulates the upstream system breaking and shipping a file
    that doesn't match the agreed schema. A schema violation should
    abort the pipeline immediately.
    """
    df = _generate_clean_rows(n_rows=500, start_user_id=2000)
    return df.drop(columns=["churned"])


def make_day3_bad_nulls() -> pd.DataFrame:
    """Day 3: same shape as clean data, but riddled with nulls.

    Simulates an upstream extraction job that partially failed -
    schema is fine, but the data is unusable.
    """
    df = _generate_clean_rows(n_rows=500, start_user_id=3000)

    # Blow out null fractions on several columns above the 10% threshold
    for col in ["monthly_spend", "avg_session_minutes", "support_tickets"]:
        idx = df.sample(frac=0.30, random_state=7).index
        df.loc[idx, col] = np.nan

    return df


def make_day4_mixed() -> pd.DataFrame:
    """Day 4: mostly clean, with a small number of bad rows mixed in.

    This is the most realistic failure mode in production - the file
    is *usable*, just imperfect. Bad rows should be filtered out and
    the remainder should pass validation.
    """
    df = _generate_clean_rows(n_rows=500, start_user_id=4000)

    # Inject a handful of corrupted rows (well under the 20% bad-row limit)
    bad_idx = df.sample(n=40, random_state=11).index

    # Negative values (impossible)
    df.loc[bad_idx[:10], "total_sessions"] = -5
    # Out-of-range subscription tier
    df.loc[bad_idx[10:20], "subscription_tier"] = "platinum"
    # Invalid churn label
    df.loc[bad_idx[20:30], "churned"] = 9
    # Wildly out-of-range spend
    df.loc[bad_idx[30:40], "monthly_spend"] = 999_999

    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "user_activity_2024_01_15.csv": ("CLEAN",   make_day1_clean()),
        "user_activity_2024_01_16.csv": ("BAD",     make_day2_bad_schema()),
        "user_activity_2024_01_17.csv": ("BAD",     make_day3_bad_nulls()),
        "user_activity_2024_01_18.csv": ("MIXED",   make_day4_mixed()),
    }

    print(f"Writing CSVs to: {RAW_DIR}\n")
    for filename, (tag, df) in files.items():
        out_path = RAW_DIR / filename
        df.to_csv(out_path, index=False)
        print(f"  [{tag:5s}] {filename}  ({len(df)} rows, {len(df.columns)} cols)")

    print("\nDone. Now run: python src/pipeline.py")


if __name__ == "__main__":
    main()
