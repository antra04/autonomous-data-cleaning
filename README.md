# Autonomous Data Cleaning and ML Benchmarking System

This project is a complete end-to-end framework for automated data cleaning, anomaly detection, dataset validation, and machine-learning model benchmarking.
It transforms raw datasets into ML-ready data and provides training, evaluation, and analytics through an interactive Streamlit dashboard.

---

## Overview

The system provides:

* Automated data cleaning
* Chunk-based large dataset handling
* Data validation and quality checks
* Statistical and ML-based anomaly detection
* Full ML model benchmarking suite
* PCA-based structure detection and clustering
* Automated report generation (TXT and PDF)
* Streamlit dashboard for cleaning, training, prediction, and visualization

This project demonstrates a production-style ML pipeline with modular code, large-file compatibility, and extendable components.

---

## Features

### 1. Data Cleaning

* Missing value imputation (median/mode)
* Outlier clipping using percentile ranges
* Type normalization
* Duplicate detection and removal
* Schema validation
* Chunk-based processing for large CSV files
* Cleaned file export to the outputs directory

### 2. Data Validation

* Missingness checks
* Schema consistency
* Numerical range analysis
* Duplicate count analysis
* Drift detection on numeric features

### 3. Anomaly Detection

Implemented using multiple approaches:

* Z-Score statistical method
* Isolation Forest algorithm
* Autoencoder-based anomaly detection

### 4. Machine-Learning Benchmarking

Evaluated multiple models:

* Logistic Regression
* RandomForestClassifier
* XGBoost
* LightGBM
* Random Search tuned variants
* Stacking Ensemble (XGBoost + LightGBM + RandomForest)

Metrics used:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrices
* Cross-validation

### 5. PCA and Clustering

* PCA for dimensionality reduction
* Visualization of 2D components
* KMeans clustering on PCA embeddings

### 6. Streamlit Dashboard

Interactive modules:

* Data Cleaning
* Dataset Preview and Diagnostics
* Model Training
* Prediction Engine
* PCA/Clustering Analysis
* Exportable reports

Run with:

```
streamlit run app/streamlit_app.py
```

---

## Model Benchmarking Results

| Model                          | CV Accuracy | Holdout Accuracy | F1 Score | ROC-AUC |
| ------------------------------ | ----------- | ---------------- | -------- | ------- |
| Logistic Regression (baseline) | 0.820       | 0.798            | 0.723    | 0.844   |
| Random Forest                  | 0.825       | 0.821            | 0.754    | 0.844   |
| XGBoost                        | 0.831       | 0.810            | 0.742    | 0.849   |
| LightGBM                       | 0.816       | 0.798            | 0.735    | 0.830   |
| XGBoost (Random Search)        | 0.834       | 0.815            | 0.744    | 0.846   |
| LightGBM (Random Search)       | 0.816       | 0.821            | 0.742    | 0.855   |
| Stacking Ensemble              | —           | 0.832            | 0.794    | —       |

### Best Model

The best model was the **Stacking Ensemble** with:

* Holdout Accuracy: 0.832
* F1 Score: 0.794

---

## Model Comparison Insights

* Tree-based models outperformed linear methods, indicating non-linear feature interactions.
* XGBoost showed strong generalization after hyperparameter tuning.
* LightGBM produced the highest ROC-AUC (0.855), indicating strong ranking capability.
* The Stacking Ensemble achieved the best accuracy and F1 score, balancing variance and bias.
* Ensemble learning significantly stabilized predictions compared to individual models.

---

## Tech Stack

* Python
* Streamlit
* Pandas, NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Plotly
* FPDF

---

## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
AutonomousDataCleaning/
│── app/
│   └── streamlit_app.py            # Main Streamlit dashboard
│
│── src/
│   ├── cleaning/                   # Cleaning utilities
│   ├── detection/                  # Anomaly detection modules
│   ├── reporting/                  # Report generation
│   └── validation/                 # Data validation logic
│
│── notebooks/
│   ├── Cleaning.ipynb
│   ├── Anomaly_detection.ipynb
│   ├── Model_benchmark.ipynb
│   └── Validation_unit_tests.ipynb
│
│── outputs/
│   ├── cleaned CSVs
│   └── saved ML models
│
│── reports/
│   ├── text reports
│   ├── PDF reports
│   └── evaluation plots
│
│── requirements.txt
│── README.md
│── .gitignore
```

---

## Skills Demonstrated

* End-to-end ML pipeline design
* Automated data quality engineering
* Anomaly detection using statistical and ML models
* Model training and evaluation
* Streamlit dashboard development
* Handling large datasets using streaming
* Project structuring and modular architecture
* Debugging complex Python environments


