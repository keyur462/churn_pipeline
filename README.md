# Churn Prediction Data Pipeline
This project is a machine learning pipeline built for a churn prediction use case. The goal is to simulate how data arrives daily, gets processed, and only triggers model training if the data quality is good enough.

Instead of focusing on complex modeling, the main idea here is to show how a real-world pipeline should behave when dealing with raw data.

## What this project does
Every time the pipeline runs:
1. It reads a CSV file (simulating daily incoming data)
2. Cleans the data (fixes small issues like missing values or incorrect types)
3. Validates the data (checks if the data is usable)
4. If validation passes → trains a model
5. If validation fails → stops and moves the file to a failed folder

## Folder layout
churn_pipeline/
├── data/
│   ├── raw/         input CSVs go here
│   ├── processed/   cleaned data (parquet)
│   └── failed/      files that didn't pass validation
├── models/          trained models (.joblib)
├── logs/            daily log files
├── src/             python code
├── requirements.txt
└── README.md

## How to run

pip install -r requirements.txt
python src/generate_data.py
python src/pipeline.py

`generate_data.py` only needs to run once - it makes the 4 sample CSVs.

## Cleaning vs Validation
These two steps are handled separately:
* Cleaning:

  * Fixes missing values
  * Converts data types
  * Removes duplicates
  * Drops obviously incorrect rows

* Validation:

  * Checks required columns exist
  * Ensures target values are valid
  * Checks null percentage
  * Ensures enough data remains after cleaning

If validation fails, the pipeline stops and the file is moved to the failed folder.

## The 4 sample files
I made 4 files to test different scenarios:

| File   | What it tests                                      | Result   |
|--------|----------------------------------------------------|----------|
| Day 15 | Clean data                                         | Trains   |
| Day 16 | Missing the `churned` column                       | Blocked  |
| Day 17 | Too many nulls (30%) in 3 columns                  | Blocked  |
| Day 18 | Some bad rows mixed in with clean ones             | Trains   |

## Validation rules

All rules are in `src/config.py` so they're easy to change. Right now I'm checking:

- All 9 columns must be present
- At least 50 rows must survive cleaning
- No column more than 10% null
- `subscription_tier` has to be free, basic, premium, or enterprise
- `churned` has to be 0 or 1
- Cleaning can't drop more than 20% of rows

If any check fails, the file is rejected and training doesn't run.

## Files in src/

- `config.py` - all settings and thresholds
- `logger.py` - logging setup, used by every other file
- `generate_data.py` - creates the 4 sample CSVs
- `ingest.py` - finds and loads files
- `clean.py` - cleaning logic
- `validate.py` - validation rules and the pass/fail decision
- `train.py` - random forest model training
- `pipeline.py` - main file, ties everything together

## Why I made certain choices

**Synthetic data instead of a real dataset.** I wanted control over the failure cases. With real data I can't reliably make a "missing column" or "30% nulls" scenario happen on demand.

**Cleaning and validation are separate.** Cleaning fixes things, validation judges. If I mix them, I can't tell whether a file got rejected because the data was bad or because cleaning was too aggressive.

**Failed files get moved, not deleted.** If something fails at 2am, I want to be able to look at the file the next morning.

**Random Forest for the model.** It's a small dataset and the focus here is the pipeline, not squeezing out the last bit of accuracy. Random Forest handles mixed data types and gives reasonable results without much tuning.


