# Churn Prediction Data Pipeline

A simple ML data pipeline for a churn prediction model. A new CSV of user activity comes in each day, and the pipeline ingests it, cleans it, validates it, and trains a model — but only if the data quality is good enough.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458.svg)

## Overview

The pipeline runs in 4 stages:

1. **Ingest** - picks up CSV files from `data/raw/`
2. **Clean** - fixes types, drops duplicates, removes obviously broken rows
3. **Validate** - checks the cleaned data against a set of quality rules
4. **Train** - runs only if validation passes; otherwise the file is quarantined

If validation fails, training is skipped and the bad file is moved to `data/failed/` with a reason prefix so it can be inspected later.

## Demo

📹 **[Watch the walkthrough video](YOUR_GOOGLE_DRIVE_LINK_HERE)**

After running the pipeline on 4 sample CSVs, you get this:
```
PIPELINE RUN SUMMARY
======================================================================
  TRAINED   user_activity_2024_01_15.csv  (accuracy=0.980)
  BLOCKED   user_activity_2024_01_16.csv  errors=1
  BLOCKED   user_activity_2024_01_17.csv  errors=3
  TRAINED   user_activity_2024_01_18.csv  (accuracy=0.967)

2/4 files triggered training.
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/<your-username>/churn-pipeline.git
cd churn-pipeline

# Install dependencies
pip install -r requirements.txt

# Generate the 4 sample CSVs (run once)
python src/generate_data.py

# Run the pipeline
python src/pipeline.py
```

## Project Structure

```
churn_pipeline/
├── data/
│   ├── raw/         input CSVs go here
│   ├── processed/   cleaned data (parquet)
│   └── failed/      files that didn't pass validation
├── models/          trained models (.joblib)
├── logs/            daily log files
├── src/
│   ├── config.py        all settings and thresholds
│   ├── logger.py        shared logging setup
│   ├── generate_data.py creates the 4 sample CSVs
│   ├── ingest.py        loads files from raw/
│   ├── clean.py         cleaning logic
│   ├── validate.py      the validation gate
│   ├── train.py         random forest training
│   └── pipeline.py      main orchestrator (run this)
├── requirements.txt
└── README.md
```

## The 4 Sample Files

I generated 4 files to test different scenarios:

| File   | What it tests                                | Expected Result |
|--------|----------------------------------------------|-----------------|
| Day 15 | Clean data                                   | Trains          |
| Day 16 | Missing the `churned` column                 | Blocked         |
| Day 17 | 30% nulls in 3 columns                       | Blocked         |
| Day 18 | Some bad rows mixed in with clean ones       | Trains          |

Day 18 is the interesting one — cleaning removes the bad rows (negative session counts, fake subscription tiers, invalid churn labels), and the remaining 460 rows pass validation.

## Pipeline Flow

```
       data/raw/*.csv
            |
            v
        [ingest]
            |
            v
        [clean]   <- fix types, drop bad rows
            |
            v
       [validate]  <- the gate
            |
        passed?
        /     \
      yes      no
       |        |
       v        v
   [train]  [quarantine]
       |        |
       v        v
   models/   data/failed/
```

## Validation Rules

All thresholds live in `src/config.py` so they're easy to tune.

| Check        | Rule                                                       |
|--------------|------------------------------------------------------------|
| Schema       | All 9 expected columns must be present                     |
| Volume       | At least 50 rows must survive cleaning                     |
| Nulls        | No column may be more than 10% null                        |
| Categoricals | `subscription_tier` must be one of free/basic/premium/enterprise; `churned` must be 0 or 1 |
| Drop rate    | Cleaning may not drop more than 20% of rows                |

## How to Verify the Output

After running the pipeline, check these folders:

```bash
dir models           # 2 .joblib files (trained models)
dir data\processed   # 2 .parquet files (cleaned data)
dir data\failed      # 2 quarantined CSVs with reason prefix
dir logs             # daily log file
```

To inspect the cleaned data:

```python
import pandas as pd
df = pd.read_parquet("data/processed/user_activity_2024_01_15.parquet")
print(df.shape)
print(df.head())
```

## Design Decisions

**Synthetic data instead of a real dataset.** I wanted control over the failure cases. With a real dataset, I can't reliably make a "missing column" or "30% nulls" scenario happen on demand.

**Cleaning and validation are separate.** Cleaning fixes things, validation judges. If I mix them, I can't tell whether a file got rejected because the data was bad or because cleaning was too aggressive.

**Failed files are moved, not deleted.** If something fails at 2am, I want to be able to look at the file the next morning.

**Random Forest for the model.** Small dataset, focus is on the pipeline. Random Forest handles mixed data types and gives reasonable results without much tuning.

## Scaling This Up

For a production setup I'd:

- Wrap `pipeline.py` in an Airflow or Prefect DAG for scheduling
- Read CSVs from S3 instead of a local folder
- Use MLflow for model tracking instead of just dumping joblib files
- Add Slack alerts when validation fails
- Add data drift checks (compare today's distributions to last week)

The stage functions wouldn't change much — only the orchestration around them.

## Tech Stack

- **Python 3.10+**
- **pandas** for data manipulation
- **scikit-learn** for the Random Forest model
- **PyArrow** for parquet storage
- **joblib** for model serialization

## Future Improvements

- Drift detection
- Great Expectations or Pandera for declarative validation
- A small dashboard for run history
- Unit tests for each stage

## Author

Keyur — built as part of an ML engineering assessment.
