import joblib
import pandas as pd

def predict_batch():
    pipeline = joblib.load('models/vehicle_valuation_pipeline.joblib')
    df = pd.read_csv('data/vehicle_inference.csv')
    X = df.drop(columns=['vehicle_id'])
    
    preds = pipeline.predict(X)
    df['estimated_fair_price'] = preds.round(-2)
    df['lower_bound_90pct'] = (preds * 0.93).round(-2)
    df['upper_bound_90pct'] = (preds * 1.07).round(-2)

    df[['vehicle_id', 'brand', 'age_years', 'odometer_km', 'estimated_fair_price', 'lower_bound_90pct', 'upper_bound_90pct']].to_csv('data/scored_vehicles.csv', index=False)
    print("Batch valuation complete -> data/scored_vehicles.csv")
    print(df[['vehicle_id', 'brand', 'estimated_fair_price']].head())

if __name__ == '__main__':
    predict_batch()
