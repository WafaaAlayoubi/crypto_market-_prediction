# Crypto Market Price Prediction using XGBoost

## Project Overview

This project focuses on predicting cryptocurrency market price movements using the powerful XGBoost regression model. Leveraging a large set of engineered features from market order book data and trade volumes, the model aims to accurately forecast the target variable (`label`) with strong performance.

---

## Dataset Description

- Contains 896 features including core market quantities (`bid_qty`, `ask_qty`, `buy_qty`, `sell_qty`, `volume`) and 890 engineered features (`X1` to `X890`).
- Training data indexed by timestamp; test data aligned with the same features but missing target labels.

---

## Feature Engineering

- Lag features capturing temporal dependencies on core features.
- Rolling window statistics (mean, std) over multiple window sizes (3, 5, 10).
- Ratio and delta features to measure market pressure and imbalance.
- Interaction terms combining volume and bid/ask or buy/sell quantities.

---

## Model: XGBoost Regressor

- Utilized XGBoost for its robustness and scalability on high-dimensional data.
- Trained on the engineered dataset to predict continuous target values.
- Early stopping and validation split used to prevent overfitting.

---

## Performance Metrics

- **R² (Coefficient of Determination): 0.9596** — Indicates that approximately 96% of the variance in the target is explained by the model.
- **RMSE (Root Mean Square Error): 0.1479** — Demonstrates high prediction accuracy with low average error magnitude.
- Pearson Correlation also calculated to confirm strong predictive relationships.

---

## How to Run

1. Install dependencies:

```bash
pip install xgboost pandas numpy scikit-learn
