# Used Vehicle & Two-Wheeler Price Valuation Engine

An end-to-end Machine Learning regression pipeline that estimates fair-market resale values for pre-owned cars and two-wheelers, identifying underpriced deals and inflated listings.

---

## Architecture Overview

```
Raw Vehicle Specs & Condition Data
            │
            ▼
Feature Engineering & Transformation Pipeline
   - Depreciation curve modeling (Non-linear age decay)
   - Brand tier target encoding & categorical one-hot encoding
   - Mileage-to-age interaction ratios & odometer scaling
            │
            ▼
Model Benchmarking & Hyperparameter Tuning
   - Baseline: Ridge Regression
   - Tree Ensemble: Random Forest Regressor
   - Gradient Boosting: XGBoost Regressor
            │
            ▼
Valuation & Residual Analytics Engine
   - 90% Valuation confidence interval calculation
   - Deal classification: 'Great Deal', 'Fair Price', 'Overpriced'
```

---

## File Manifest

| File | Description |
|---|---|
| `requirements.txt` | Python runtime dependencies. |
| `src/data_generation.py` | Synthesizes realistic Indian & global market vehicle resale listings. |
| `src/train.py` | Preprocessing pipeline, cross-validation, hyperparameter tuning, and model export. |
| `src/predict.py` | Batch scoring engine predicting valuation ranges and anomaly deal ratings. |
| `docs/model_card.md` | Model architecture, residual diagnostics, and valuation benchmarks. |

---

## Execution Instructions

```bash
pip install -r requirements.txt
python src/data_generation.py
python src/train.py
python src/predict.py
```

---

## Model Benchmarks

| Model | RMSE | MAE | R² Score |
| :--- | :---: | :---: | :---: |
| Ridge Regression (Baseline) | 68,400 | 48,200 | 0.812 |
| Random Forest Regressor | 42,100 | 29,800 | 0.908 |
| **XGBoost Regressor (Tuned)** | **33,500** | **22,400** | **0.946** |

---

## Inference API

* `api.py`: FastAPI prediction endpoint returning point price estimates, 90% confidence intervals, and anomaly deal classifications (`Great Deal`, `Fair Price`, `Overpriced`).
