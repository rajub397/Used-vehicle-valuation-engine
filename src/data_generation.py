import os
import numpy as np
import pandas as pd

def generate_vehicle_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    vehicle_types = np.random.choice(['Car', 'Two-Wheeler'], size=n_samples, p=[0.60, 0.40])
    brands_car = ['Maruti Suzuki', 'Hyundai', 'Honda', 'Toyota', 'Tata', 'BMW', 'Mercedes-Benz']
    brands_bike = ['Hero', 'Honda', 'Bajaj', 'TVS', 'Royal Enfield', 'KTM', 'Yamaha']
    
    brand, brand_tier, original_price = [], [], []
    for vt in vehicle_types:
        if vt == 'Car':
            b = np.random.choice(brands_car, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.05, 0.05])
            brand.append(b)
            if b in ['BMW', 'Mercedes-Benz']:
                brand_tier.append('Luxury')
                original_price.append(np.random.uniform(4000000, 7500000))
            elif b in ['Toyota', 'Honda']:
                brand_tier.append('Premium')
                original_price.append(np.random.uniform(1200000, 2500000))
            else:
                brand_tier.append('Economy')
                original_price.append(np.random.uniform(600000, 1400000))
        else:
            b = np.random.choice(brands_bike, p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05])
            brand.append(b)
            if b in ['Royal Enfield', 'KTM']:
                brand_tier.append('Premium')
                original_price.append(np.random.uniform(180000, 320000))
            else:
                brand_tier.append('Economy')
                original_price.append(np.random.uniform(70000, 140000))
                
    age_years = np.random.uniform(0.5, 12, size=n_samples).round(1)
    km_per_year = np.random.normal(loc=12000, scale=3500, size=n_samples).clip(2000, 35000)
    odometer_km = (age_years * km_per_year).round()
    
    fuel_types = np.random.choice(['Petrol', 'Diesel', 'CNG', 'Electric'], size=n_samples, p=[0.55, 0.30, 0.10, 0.05])
    owners = np.random.choice([1, 2, 3], size=n_samples, p=[0.70, 0.24, 0.06])
    service_history = np.random.choice(['Authorized Dealership', 'Independent Garage', 'Irregular'], size=n_samples, p=[0.55, 0.35, 0.10])
    
    # Valuation logic
    depreciation = (
        0.12 * age_years +
        0.0000025 * odometer_km +
        0.08 * (owners - 1) +
        0.07 * (service_history == 'Irregular').astype(int) -
        0.04 * int(brand_tier == 'Luxury')
    )
    depreciation = np.clip(depreciation, 0.15, 0.82)
    resale_price = (np.array(original_price) * (1 - depreciation) * np.random.uniform(0.94, 1.06, size=n_samples)).round(-2)
    
    df = pd.DataFrame({
        'vehicle_id': [f"VEH-{10000 + i}" for i in range(n_samples)],
        'vehicle_type': vehicle_types,
        'brand': brand,
        'brand_tier': brand_tier,
        'original_price': np.array(original_price).round(-2),
        'age_years': age_years,
        'odometer_km': odometer_km,
        'fuel_type': fuel_types,
        'owner_count': owners,
        'service_history': service_history,
        'resale_price': resale_price
    })
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_vehicle_data(6000)
    train_df = df.iloc[:5000]
    holdout_df = df.iloc[5000:].drop(columns=['resale_price'])
    train_df.to_csv('data/vehicle_train.csv', index=False)
    holdout_df.to_csv('data/vehicle_inference.csv', index=False)
    print("Generated 5,000 train rows -> data/vehicle_train.csv")
    print("Generated 1,000 inference rows -> data/vehicle_inference.csv")
