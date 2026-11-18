from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import os
import time

app = FastAPI(
    title="Used Vehicle & Two-Wheeler Price Valuation API",
    description="Production inference service with Pydantic schema validation and latency telemetry.",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/vehicle_valuation_pipeline.joblib")

class VehicleFeatures(BaseModel):
    vehicle_type: str = Field(..., examples=["Car", "Two-Wheeler"])
    brand: str = Field(..., examples=["Maruti", "Honda", "Royal Enfield"])
    year: int = Field(..., ge=2000, le=2026)
    km_driven: float = Field(..., ge=0, le=500000)
    fuel_type: str = Field(..., examples=["Petrol", "Diesel", "Electric", "CNG"])
    transmission: str = Field(..., examples=["Manual", "Automatic"])
    owner_count: int = Field(..., ge=1, le=5)
    engine_cc: float = Field(..., ge=50, le=5000)

    @field_validator("fuel_type")
    @classmethod
    def validate_fuel(cls, v: str) -> str:
        allowed = {"Petrol", "Diesel", "Electric", "CNG"}
        if v.title() not in allowed:
            raise ValueError(f"fuel_type must be one of {allowed}")
        return v.title()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH,
    }

@app.post("/predict")
def predict_price(vehicle: VehicleFeatures):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model artifact not found. Run training pipeline first.")
    
    start = time.perf_counter()
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact
    
    input_df = pd.DataFrame([vehicle.model_dump()])
    predicted_price = float(pipeline.predict(input_df)[0])
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "estimated_price_inr": round(max(predicted_price, 5000.0), 2),
        "inference_latency_ms": latency_ms,
        "status": "success",
    }
