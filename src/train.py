import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

def train():
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('data/vehicle_train.csv')
    X = df.drop(columns=['vehicle_id', 'resale_price'])
    y = df['resale_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_cols = ['original_price', 'age_years', 'odometer_km', 'owner_count']
    cat_cols = ['vehicle_type', 'brand', 'brand_tier', 'fuel_type', 'service_history']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Validation MAE:  {mae:.2f}")
    print(f"Validation R²:   {r2:.4f}")

    joblib.dump(pipeline, 'models/vehicle_valuation_pipeline.joblib')
    print("Saved pipeline to models/vehicle_valuation_pipeline.joblib")

if __name__ == '__main__':
    train()
