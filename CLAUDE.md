# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA game outcome prediction data pipeline. Fetches raw gamelogs/schedules from S3, engineers 63+ features, validates output with Pydantic schemas, and produces training/in-season datasets. Orchestrated by DVC (Data Version Control).

## Commands

```bash
# Install dependencies
poetry install

# Run full pipeline
poetry run dvc repro -f

# Run pipeline up to a specific stage
poetry run dvc repro post_cleaning_dataset -f

# Run all tests
poetry run pytest

# Run a single test file
poetry run pytest tests/test_feature_engineering_pipeline.py

# Run a single test method
poetry run pytest tests/test_feature_engineering_pipeline.py::TestFeatureEngineeringPipeline::test_method_name

# Lint
poetry run pylint src/

# Format
poetry run black src/ tests/

# Check test coverage
poetry run pytest --cov=src --cov-report=html
```

## Pipeline Architecture (DVC stages in order)

1. **get_training_dataset** → Fetch gamelogs + schedules from S3
2. **gamelog_schedule_unification** → Merge into unified dataset
3. **pre_cleaning_dataset** → Deduplicate, remove COVID bubble games, remove playoffs (game_nb > 82), filter to current date
4. **features_engineering_pipeline** → Apply 13+ feature engineering functions (rolling averages, win ratios, streaks, rest days, travel time, opponent stats, etc.)
5. **get_y_variables** → Generate target/prediction variables
6. **post_cleaning_dataset** → Remove game 1 (no prior features), split into training set (historical) and in-season set (current 2026 season)
7. **training_dataset_assessment** → Validate output with Pydantic schemas (pandantic)
8. **writte_final_output_to_s3** → Upload final CSVs to S3

## Key Architecture Patterns

- **Feature engineering pipeline** (`src/features_engineering_pipeline.py`) chains 13+ functions using pandas `.pipe()`. Each function lives in its own module under `src/feature_engineering_functions/`.
- **Data is sorted by `(id_season, tm, game_nb)`** before feature engineering. Rolling/expanding window calculations depend on this sort order.
- **Opponent features** are computed by merging a team's features onto its opponent's row — this doubles many columns with `_opp` suffix.
- **Pipeline parameters** come from `params.yaml` (data paths, file names) and `.env` (AWS/MySQL credentials).
- **Data flows through `data/` subdirectories:** `raw/` → `input/` → `processed/` → `output/`
- Tests use `unittest.TestCase` style and some read from `data/processed/` (real pipeline output).

## Environment Requirements

- Python 3.11+
- Poetry for dependency management
- AWS credentials in `.env` for S3 access (region: eu-central-1)
- MySQL credentials in `.env` for database access

## pytest Configuration

pytest is configured in `pyproject.toml` with `importmode = "importlib"` and `pythonpath = "."`.
