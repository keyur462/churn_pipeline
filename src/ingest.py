from pathlib import Path
from typing import List

import pandas as pd

from config import RAW_DIR
from logger import get_logger

log = get_logger("ingest")


def list_raw_files() -> List[Path]:
    """Return all CSVs in the raw directory, sorted by name (so dates come in order)."""
    files = sorted(RAW_DIR.glob("*.csv"))
    log.info(f"Found {len(files)} raw file(s) in {RAW_DIR}")
    return files


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV into a DataFrame. Raises if the file is unreadable."""
    log.info(f"Loading {path.name}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.error(f"Failed to read {path.name}: {e}")
        raise
    log.info(f"Loaded {path.name}: {len(df)} rows, {len(df.columns)} columns")
    return df
