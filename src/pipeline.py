import shutil
import sys
from pathlib import Path

from config import FAILED_DIR, PROCESSED_DIR
from ingest import list_raw_files, load_csv
from clean import clean
from validate import validate
from train import train
from logger import get_logger

log = get_logger("pipeline")


def process_file(path: Path) -> dict:
    """Run the full pipeline on a single CSV. Returns a summary dict."""
    log.info("=" * 70)
    log.info(f"Processing: {path.name}")
    log.info("=" * 70)

    summary = {
        "file":             path.name,
        "validation_passed": False,
        "training_triggered": False,
        "errors":           [],
    }

    # ---- Stage 1: Ingest ----
    try:
        df = load_csv(path)
    except Exception as e:
        summary["errors"].append(f"Ingest failed: {e}")
        _quarantine(path, reason="unreadable")
        return summary

    # ---- Stage 2: Clean ----
    df_clean, clean_stats = clean(df)

    # ---- Stage 3: Validate (the gate) ----
    result = validate(df_clean, clean_stats)
    summary["validation_passed"] = result.passed
    summary["errors"] = result.errors

    if not result.passed:
        log.error(f"Training BLOCKED for {path.name}. Moving to {FAILED_DIR}.")
        _quarantine(path, reason="validation_failed")
        return summary

    # ---- Persist the cleaned data ----
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / path.name.replace(".csv", ".parquet")
    df_clean.to_parquet(processed_path, index=False)
    log.info(f"Processed data saved to {processed_path}")

    # ---- Stage 4: Train (conditional) ----
    log.info(f"Validation passed. Triggering training for {path.name}.")
    metrics = train(df_clean, source_filename=path.name)
    summary["training_triggered"] = True
    summary["accuracy"] = metrics["accuracy"]
    summary["model_path"] = metrics["model_path"]

    return summary


def _quarantine(src: Path, reason: str) -> None:
    """Move a failed file out of raw/ so we don't reprocess it next run."""
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    dst = FAILED_DIR / f"{reason}__{src.name}"
    shutil.move(str(src), str(dst))
    log.info(f"Quarantined: {dst}")


def _archive_processed(src: Path) -> None:
    """Move successfully-processed file out of raw/ to keep next run clean."""
    archive_dir = src.parent / "_archive"
    archive_dir.mkdir(exist_ok=True)
    shutil.move(str(src), str(archive_dir / src.name))


def main() -> int:
    files = list_raw_files()
    if not files:
        log.warning("No CSV files found in data/raw/. Run generate_data.py first.")
        return 1

    summaries = []
    for f in files:
        try:
            summary = process_file(f)
            summaries.append(summary)
            # Archive only the successful ones; failed ones already moved
            if summary["validation_passed"]:
                _archive_processed(f)
        except Exception as e:
            log.exception(f"Unhandled error processing {f.name}: {e}")
            summaries.append({"file": f.name, "errors": [str(e)]})

    # ---- Final summary ----
    log.info("=" * 70)
    log.info("PIPELINE RUN SUMMARY")
    log.info("=" * 70)
    for s in summaries:
        status = "TRAINED" if s.get("training_triggered") else "BLOCKED"
        line = f"  {status:8s}  {s['file']}"
        if "accuracy" in s:
            line += f"  (accuracy={s['accuracy']:.3f})"
        if s.get("errors"):
            line += f"  errors={len(s['errors'])}"
        log.info(line)

    n_trained = sum(1 for s in summaries if s.get("training_triggered"))
    log.info(f"\n{n_trained}/{len(summaries)} files triggered training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
