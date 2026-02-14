
# Retail Demand Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?logo=TensorFlow&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-S3-orange)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?logo=Apache%20Airflow&logoColor=white)

## Overview

This project implements an end-to-end demand-prediction pipeline designed to optimize inventory management and sales strategies for retail businesses. By integrating historical sales data with external factors such as promotions, holidays, and weather patterns, the system generates highly accurate forecasts at both store and item levels.

The core of the project moves beyond classical statistical methods, utilizing Deep Learning architectures—specifically **Long Short-Term Memory (LSTM)** networks and **Temporal Fusion Transformers (TFT)**—to capture complex temporal dependencies.

> **Key Achievement:** The implementation of TFT and LSTM models achieved a **22% improvement in forecast accuracy** compared to classical ARIMA baselines.

## Key Features

- **Advanced Modeling:** Implemented LSTM and Temporal Fusion Transformers (TFT) for time-series forecasting.
- **Rich Feature Engineering:** Integrates diverse datasets including:
  - Historical Sales Data
  - Promotional Events
  - Holiday Calendars
  - Weather Conditions
- **MLOps Pipeline:**
  - **Data Lake:** Utilizes **AWS S3** for scalable data storage.
  - **Orchestration:** Automated pipeline execution using **Apache Airflow**.
  - **Experiment Tracking:** Model versioning and metric tracking via **MLflow**.
- **Scalability:** Extensible architecture allowing for the easy addition of new prediction methods or feature sets.

## Tech Stack

| Category | Tools Used |
| :--- | :--- |
| **Languages** | Python (3.8+) |
| **Deep Learning** | TensorFlow, PyTorch |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **MLOps & Cloud** | Apache Airflow, MLflow, AWS S3 |

## Architecture & Workflow

The pipeline follows a modular architecture:

1.  **Ingestion:** Raw data (sales, weather, holidays) is ingested from AWS S3.
2.  **Preprocessing:** Data cleaning, normalization, and feature merging (holidays/weather).
3.  **Training:** Models (LSTM/TFT) are trained on historical sequences.
4.  **Evaluation:** Performance is logged to MLflow; best models are selected based on RMSE/MAE.
5.  **Forecasting:** Future demand is predicted and saved to the submissions directory.

## Installation

### Prerequisites

- Python 3.8+
- Required libraries (see `requirements.txt`)

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/retail-forecasting.git](https://github.com/your-username/retail-forecasting.git)
    cd retail-forecasting
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment:**
    Create a `.env` file to store your AWS credentials and MLflow tracking URI (if running remotely).

## Usage

### Running the Forecasting Script

You can run the pipeline via the command line, specifying the model type and data source.

**Basic Usage:**
```bash
python main.py --data ./data --method avg --output ./submissions/forecast.csv

```

**Running the Advanced TFT Model:**

```bash
python main.py --data ./data --method tft --output ./submissions/forecast_tft.csv

```

### Command-line Arguments

* `--data`: Path to the data directory (default: 'data')
* `--method`: Forecasting method to use.
* Options: `avg` (baseline), `arima`, `lstm`, `tft`


* `--output`: Path to save the submission file

## Project Structure

```
retail-forecasting/
│
├── main.py              # Entry point for the forecasting pipeline
├── models/              # Model definitions (LSTM, TFT, ARIMA)
├── util/
│   ├── preprocess.py    # Feature engineering (weather/holidays integration)
│   └── aws_utils.py     # S3 connection helpers
├── data/                # Local data directory (synced with S3)
└── submissions/         # Generated forecast outputs

```

## Performance & Kaggle History

This solution was developed in part for the [Kaggle ML Zoomcamp Competition](https://www.kaggle.com/competitions/ml-zoomcamp-2024-competition).

| Date | Method / Experiment | Score (RMSE) | Improvement Notes |
| --- | --- | --- | --- |
| 2024-12-19 | Baseline (Feature Engineering) | 0.13567 | Initial baseline |
| 2024-12-20 | Ensemble (XGBoost + LightGBM) | 0.14567 | Tree-based tests |
| **2024-12-25** | **Temporal Fusion Transformer (TFT)** | **0.10542** | **22% improvement over baseline** |

## License

Distributed under the MIT License. See `LICENSE` for more information.

```

```# Retail-Demand-Forecasting.
