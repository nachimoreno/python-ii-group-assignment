# ML Stock Pricing Pipeline

A Python project for ingesting market data, preparing company-level datasets, engineering predictive features, training XGBoost models, and evaluating a simple trading strategy through a single orchestration entrypoint: `manage.py`.

This repository is organized as a sequential pipeline. At a high level, it:

1. downloads raw market data,
2. filters and cleans it for the configured companies,
3. engineers model-ready features and targets,
4. trains classification and regression models per company, and
5. runs a holdout-period trading/backtesting step from the trained models.

## Pipeline overview

The project is split into five executable stages:

### 1. Data ingestion

`data_ingestion.py` loads configuration, reads your SimFin API key from the environment, downloads U.S. company metadata plus daily share-price data, and writes raw parquet files into the bronze layer. Its main outputs are:

- `BRONZE_DIR/parquet/companies.parquet`
- `BRONZE_DIR/parquet/share_prices.parquet`

### 2. Data cleaning

`data_cleaning.py` reads the raw parquet files, selects only the companies defined in `src/config.toml`, removes unneeded columns, and writes one cleaned parquet file per configured company into the silver layer.

### 3. Feature engineering

`feature_engineering.py` reads each cleaned company dataset and builds model-ready features such as returns, rolling volatility, moving-average style signals, momentum, volume features, interaction terms, and next-period targets. The outputs are written as modeled parquet files in the gold layer.

### 4. Model training

`model_training.py` trains two XGBoost models per company:

- a classifier for next-period direction, and
- a regressor for next-period return magnitude.

It uses a chronological holdout split and time-series cross-validation during tuning, then saves fitted models and metadata under the trained-models directory.

### 5. Trading logic / backtesting

`trading_logic.py` loads the trained models and engineered datasets, produces holdout-period trading signals, simulates a simple trading strategy, and saves both detailed backtest logs and summary reports.

## Why `manage.py` is the main entrypoint

`manage.py` is the project management module that orchestrates the whole workflow. It imports each pipeline module and can either:

- run the entire pipeline in order, or
- run one specific stage with `--stage`.

The execution order baked into `manage.py` is:

1. `data_ingestion`
2. `data_cleaning`
3. `feature_engineering`
4. `model_training`
5. `trading_logic`

That makes `manage.py` the recommended way to run the project.

## Requirements

The repository requires Python 3.11 and includes the following Python dependencies in `requirements.txt`:

- `pandas==2.2.3`
- `simfin`
- `polars`
- `python-dotenv`
- `xgboost`
- `scikit-learn`
- `pyarrow`
- plus supporting packages such as `charset-normalizer` and `llvm-openmp`.

## Configuration before running

Before executing the pipeline, make sure these are in place:

### 1. `src/config.toml`

This file is used across the pipeline for directory paths, and several stages also depend on a configured `companies` list.

### 2. SimFin API key

`data_ingestion.py` expects an environment variable named `API_KEY`. The easiest setup is to place it in a `.env` file at the project root:

```env
API_KEY=your_simfin_api_key_here
```

`python-dotenv` is included in the requirements, and the ingestion stage depends on that key being present and valid.

## Installation option 1: conda environment + pip dependencies

Some required libraries are installed through `pip`, so the recommended conda-based setup is to create the environment with the correct Python version first, then install the project dependencies from `requirements.txt` using `pip`.

### Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate python_ii_group_assignment
```

## Running the project with `manage.py`

### Run the full pipeline

This is the standard command for an end-to-end execution:

```bash
python manage.py --stage all
```

This will run, in order:

1. ingestion,
2. cleaning,
3. feature engineering,
4. model training,
5. trading logic / backtesting.

### Run a single stage

You can also run one stage at a time through the same entrypoint:

```bash
python manage.py --stage data_ingestion
python manage.py --stage data_cleaning
python manage.py --stage feature_engineering
python manage.py --stage model_training
python manage.py --stage trading_logic
```

This is useful when:

- you want to rerun only one part of the workflow,
- you are debugging a specific stage, or
- earlier outputs already exist and do not need to be regenerated.

## Recommended execution pattern

For a first run:

```bash
python manage.py --stage all
```

For iterative development:

- rerun `data_ingestion` only if you need fresh raw data,
- rerun `data_cleaning` or `feature_engineering` after changing preprocessing logic,
- rerun `model_training` after feature or modeling changes,
- rerun `trading_logic` after updating strategy or execution rules.

Because `manage.py` does not enforce dependency checks between stages, running a later stage assumes the required upstream outputs already exist. For example, `model_training` expects modeled gold-layer parquet files to already be available, and `trading_logic` expects trained models plus prior datasets to exist.

## Expected outputs by stage

### After `data_ingestion`

```text
BRONZE_DIR/
  parquet/
    companies.parquet
    share_prices.parquet
```

### After `data_cleaning`

```text
SILVER_DIR/
  parquet/
    {company}_share_prices_cleaned.parquet
```

### After `feature_engineering`

```text
GOLD_DIR/
  parquet/
    {company}_share_prices_modeled.parquet
```

### After `model_training`

```text
GOLD_DIR/
  trained_models/
    {company_folder}/
      classification_model.pkl
      regression_model.pkl
      metadata.json
```

### After `trading_logic`

```text
GOLD_DIR/
  backtests/
    {company_folder}/
      holdout_backtest.csv
      holdout_backtest_summary.json
    all_company_backtest_summary.csv
```

These stage outputs are described in the individual script summaries and reflect the intended pipeline handoff between bronze, silver, and gold data layers.

## Example quickstart

```bash
# 1) create environment
conda create --name ie_pfda_mlgp
conda activate ie_pfda_mlgp

# 2) install dependencies
pip install -r requirements.txt

# 3) add your SimFin API key to .env
API_KEY=your_simfin_api_key_here

# 4) verify src/config.toml is configured

# 5) run everything
python manage.py --stage all
```

## Troubleshooting notes

### Missing API key

If ingestion fails early, verify that your `.env` file exists and that `API_KEY` is set correctly. The ingestion stage depends on that environment variable for SimFin access.

### Running a later stage directly

If you run `model_training` or `trading_logic` before upstream artifacts exist, the stage will fail because `manage.py` does not build missing dependencies automatically.

### Company naming consistency

Several stages assume the configured company names in `config.toml` match the source data exactly, and multiple outputs are written per company. Keep naming consistent across configuration and generated files. Please ensure that the company names in `config.toml` are spelled exactly as they appear in the SIMFIN API endpoint.

## Summary

Use `manage.py` as the operational entrypoint for the repository. Set up Python 3.11+, install dependencies using either `.venv` or conda-plus-pip, provide a valid SimFin `API_KEY`, confirm your `src/config.toml` values, and run the project with:

```bash
python manage.py --stage all
```

When needed, use `--stage` to execute a single step in the pipeline. That keeps the workflow simple while matching the project’s intended orchestration model.
