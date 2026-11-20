# Appliance Energy Consumption Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.1+-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Development-yellow.svg)

**An end-to-end machine-learning and deep-learning system for predicting household appliance energy consumption.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Business Context](#business-context)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
- [Model Pipeline](#model-pipeline)
- [Model Results](#model-results)
- [Inference](#inference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Notebooks](#notebooks)


---

## Overview

This project predicts next-step appliance energy consumption from household sensor measurements and weather data. It combines classical machine-learning models with recurrent neural networks and provides one inference interface for comparing every model family using the same input context.

The implementation follows an executable, configuration-driven pipeline:

- Chronological train/test splitting to prevent future information leaking into training
- Missing-value interpolation and 1st/99th-percentile feature clipping
- Cyclical time features, target lag features, rolling features, and domain features
- Random Forest-based feature importance and correlation filtering
- MinMax scaling fitted only on training data
- Linear Regression, Random Forest, XGBoost, tuned ensemble models, LSTM, and GRU
- MAE, RMSE, MAPE, and R² evaluation
- Side-by-side ML and deep-learning inference

---

## Business Context

Accurate short-term appliance energy forecasts can support:

- Peak-demand monitoring
- Smart-home and building automation
- Energy-cost forecasting
- Occupancy and equipment-efficiency analysis
- Scheduling of controllable electrical loads

The target is the `Appliances` energy measurement in watt-hours. The forecasting horizon is the next 10-minute observation.

---

## Dataset

The project uses the UCI **Appliances Energy Prediction** dataset.

| Attribute | Value |
|---|---:|
| Raw records | 19,735 |
| Original columns | 29 |
| Target | `Appliances` |
| Time period | January 11, 2016 to May 27, 2016 |
| Missing values | None |
| Target minimum | 10 Wh |
| Target median | 60 Wh |
| Target mean | 97.69 Wh |
| Target maximum | 1,080 Wh |
| Training rows after feature engineering | 15,759 |
| Test rows after feature engineering | 3,940 |

The dataset contains indoor temperature and humidity sensors, outdoor weather measurements, lighting power, pressure, wind speed, visibility, and dew-point measurements.

---

## Project Structure

```text
appliance-energy-prediction/
├── artifacts/
│   ├── metrics/
│   │   ├── gru_loss_history.json
│   │   ├── lstm_loss_history.json
│   │   └── model_comparison.csv
│   ├── models/
│   │   ├── best_ml_model.pkl
│   │   ├── gru_model.pth
│   │   ├── lstm_model.pth
│   │   └── *_model.pkl
│   └── scalers/
│       ├── outlier_clipper.pkl
│       ├── selected_features.pkl
│       ├── scaler_X.pkl
│       └── scaler_y.pkl
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── energy_data_set.csv
│   └── processed/
│       ├── featured_data.csv
│       ├── train.csv
│       ├── test.csv
│       ├── X_train_scaled.npy
│       ├── X_test_scaled.npy
│       └── y_train.npy
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_handle_outlier.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_multi_model_training.ipynb
│   └── 05_dl_models.ipynb
├── pipeline/
│   ├── data_pipeline.py
│   ├── training_pipeline.py
│   └── inference_pipeline.py
├── src/
│   ├── data_ingestion.py
│   ├── outlier_handler.py
│   ├── feature_engineering.py
│   ├── deep_learning_model.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_inference.py
├── tests/
│   └── test_deep_learning_model.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | Pandas, NumPy |
| Classical ML | scikit-learn, XGBoost |
| Deep learning | PyTorch |
| Feature analysis | Random Forest, Statsmodels |
| Visualization | Matplotlib, Seaborn |
| Serialization | Joblib, PyTorch checkpoints |
| Configuration | PyYAML |
| Testing | pytest, pytest-cov |
| Development quality | Ruff, mypy |

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/nadukayomal/appliance-energy-prediction-dl.git
cd appliance-energy-prediction-dl
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

For a normal environment:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

For development tools:

```powershell
python -m pip install -e ".[dev]"
```

Alternatively, install directly from the requirements file:

```powershell
python -m pip install -r requirements.txt
```

---

## Quick Start

Run each stage from the repository root:

```powershell
python pipeline/data_pipeline.py
python pipeline/training_pipeline.py
python pipeline/inference_pipeline.py
```

Or use the project entry points after installing the package:

```powershell
run-data-pipeline
run-training-pipeline
run-inference-pipeline
```

---

## Data Pipeline

The data pipeline is implemented by [TemporalDataPipeline](/D:/portfolio%20projects/energy-consumption/pipeline/data_pipeline.py).

### Processing stages

1. Load and validate the raw CSV.
2. Sort all observations by the `date` datetime index.
3. Remove `rv1` and `rv2`.
4. Interpolate missing values chronologically.
5. Split the data in an 80/20 chronological order.
6. Clip `lights`, `Visibility`, and `RH_5` using training-only 1st/99th-percentile bounds.
7. Create cyclical hour and day-of-week features.
8. Create lag features at 1, 2, 3, 6, 12, 18, and 36 observations.
9. Create shifted rolling mean and standard-deviation features.
10. Create indoor temperature, humidity, interaction, and lighting-domain features.
11. Remove rows introduced with lag or rolling NaNs.
12. Train a Random Forest to rank features.
13. Remove highly correlated features and retain the top 25 uncorrelated features.
14. Fit MinMax scalers on training features and the target.

The preprocessed arrays and serialized preprocessing objects are written to [data/processed/](/D:/portfolio%20projects/energy-consumption/data/processed) and [artifacts/](/D:/portfolio%20projects/energy-consumption/artifacts).

---

## Model Pipeline

### Classical models

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Tuned Random Forest using `TimeSeriesSplit`
- Tuned XGBoost using `TimeSeriesSplit`

Tuned models use chronological cross-validation and randomized search with MAE as the optimization metric.

### Deep-learning models

The recurrent models in [deep_learning_model.py](/D:/portfolio%20projects/energy-consumption/src/deep_learning_model.py) use:

- Configurable LSTM or GRU architecture
- 64-unit first recurrent layer
- 32-unit second recurrent layer
- 16-unit dense layer
- 0.2 dropout
- Adam optimizer
- MSE loss
- ReduceLROnPlateau learning-rate scheduling
- Validation-based early stopping

A sequence consists of 10 observations. Each sequence predicts the next appliance-energy value.

### Evaluation metrics

- **MAE**: Mean absolute error
- **RMSE**: Root mean squared error
- **MAPE**: Mean absolute percentage error
- **R²**: Coefficient of determination

Comparison results are saved to [artifacts/metrics/model_comparison.csv](/D:/portfolio%20projects/energy-consumption/artifacts/metrics/model_comparison.csv).

---

## Model Results

The following results were produced with the implemented pipeline.

| Model | Family | MAE | RMSE | MAPE | R² |
|---|---|---:|---:|---:|---:|
| Linear Regression | ML | 26.6489 | 59.7581 | 24.2824% | 0.5639 |
| Tuned XGBoost | ML | 27.9945 | 58.7748 | 26.8234% | 0.5781 |
| Tuned Random Forest | ML | 33.8511 | 64.1337 | 33.1799% | 0.4976 |
| Random Forest | ML | 54.8820 | 88.2436 | 60.2332% | 0.0490 |
| XGBoost | ML | 66.3315 | 102.4347 | 79.8587% | -0.2815 |
| LSTM | DL | 35.7224 | 73.1417 | 34.1065% | 0.3189 |
| GRU | DL | 35.9306 | 72.9864 | 34.0495% | 0.3218 |

The verified best ML model is Linear Regression by MAE. LSTM and GRU produce competitive deep-learning baselines, but the tested classical baselines perform better on this dataset split.

---

## Inference

The [inference pipeline](/D:/portfolio%20projects/energy-consumption/pipeline/inference_pipeline.py) loads:

- The best serialized machine-learning model
- LSTM and/or GRU checkpoints
- Feature scaler
- Target scaler
- Selected feature list
- Fitted outlier clipping configuration

The interface accepts chronological raw observations ending with the row to forecast. It applies the same feature engineering and preprocessing sequence, then returns:

- ML prediction
- LSTM prediction
- GRU prediction
- Absolute differences between model outputs

Running the pipeline currently uses the final test-period context:

```powershell
python pipeline/inference_pipeline.py
```

Example result:

```json
{
  "LSTM": {
    "ML prediction": 320.0066,
    "DL prediction": 244.154,
    "Absolute difference": 75.8526
  },
  "GRU": {
    "ML prediction": 320.0066,
    "DL prediction": 212.9331,
    "Absolute difference": 107.0734
  }
}
```

Predictions are expressed in the original appliance-energy unit.

---

## Configuration

All major settings are stored in [config/config.yaml](/D:/portfolio%20projects/energy-consumption/config/config.yaml), including:

- Raw and processed data paths
- Artifact paths
- Target and datetime columns
- Temporal split settings
- Outlier clipping bounds
- Feature-selection thresholds
- Sequence length
- Baseline model names
- Deep-learning architecture settings
- Hyperparameters
- Evaluation metrics
- Reproducibility seeds

---

## Testing

Run the focused test suite:

```powershell
python -m pytest -q tests
```

Current tests verify:

- Sequence array shapes
- Evaluation metric outputs
- LSTM forward-pass tensor dimensions

---

## Notebooks

The development workflow is preserved in [notebooks/](/D:/portfolio%20projects/energy-consumption/notebooks):

| Notebook | Purpose |
|---|---|
| `01_EDA.ipynb` | Data inspection, trends, seasonality, and correlation analysis |
| `02_handle_outlier.ipynb` | IQR/Z-score analysis and training-fitted percentile clipping |
| `03_feature_engineering.ipynb` | Cyclical, lag, rolling, and domain feature engineering |
| `04_multi_model_training.ipynb` | Baseline models and tuned ensemble-model comparison |
| `05_dl_models.ipynb` | PyTorch sequence creation, LSTM training, and evaluation |

The production pipelines incorporate these notebook stages into reusable classes rather than retaining notebook-only logic.

---
