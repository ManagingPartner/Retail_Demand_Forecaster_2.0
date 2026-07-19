# Retail Demand Forecaster

A time-series pipeline for forecasting daily retail demand, using a `RandomForestRegressor` from scikit-learn to produce stable, numeric sales predictions.

## Overview

This project builds a simple, clean workflow for daily retail demand forecasting:

- Generates a time-indexed dataset with calendar-based features (day of week, weekend flag, holiday flag)
- Trains a `RandomForestRegressor` to predict daily sales
- Evaluates model performance using Mean Absolute Error (MAE)

## Project Structure

```
.
├── data_loader.py   # Data generation, feature engineering, and train/test split
├── forecaster.py    # Model training and evaluation
└── README.md
```

## How It Works

**`data_loader.py`**
- Creates 3 years (1,095 days) of daily date records starting 2023-01-01
- Engineers features: `day_of_week`, `is_weekend`, and an `is_holiday` placeholder (currently always 0 — extend this with real holiday logic as needed)
- Generates a synthetic `sales` target with weekly seasonality and random noise
- Splits the data into training (80%) and test (20%) sets

**`forecaster.py`**
- Loads the train/test data via `data_loader`
- Trains a `RandomForestRegressor` (100 estimators) on the training set
- Predicts on the test set and reports Mean Absolute Error (MAE)

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn

## Installation

```bash
pip install pandas numpy scikit-learn
```

## Usage

Run the forecaster directly:

```bash
python forecaster.py
```

Example output:

```
Model Training Complete.
Mean Absolute Error: X.XX
```

## Notes / Next Steps

- The dataset is currently **synthetic** — swap in real sales data by modifying `load_data()` in `data_loader.py`.
- The `is_holiday` feature is a placeholder; add real holiday calendar logic to capture holiday-driven demand surges.
- Additional features (e.g., promotions, month/quarter, lag features, rolling averages) could improve forecast accuracy.
- Consider adding hyperparameter tuning or cross-validation for more robust evaluation.
