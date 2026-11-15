# Model Card: Used Vehicle Valuation Engine

## Model Details
- **Task:** Multi-variate continuous regression predicting vehicle resale price in INR / USD.
- **Model:** Gradient Boosted Trees (XGBoost Regressor) wrapped inside Scikit-Learn `Pipeline`.
- **Target Variable:** `resale_price`

## Performance Metrics

| Model | RMSE | MAE | R² Score |
|---|---|---|---|
| Ridge Regression (Baseline) | 68,400 | 48,200 | 0.812 |
| Random Forest Regressor | 42,100 | 29,800 | 0.908 |
| **XGBoost Regressor (Tuned)** | **33,500** | **22,400** | **0.946** |

## Key Features
1. `vehicle_age_years`: Primary non-linear exponential depreciation driver.
2. `odometer_km`: Wear-and-tear metric interacted with vehicle age.
3. `brand_tier`: Luxury vs. Economy market price floor.
4. `fuel_type`: Premium retention on Diesel / Hybrid variants.
