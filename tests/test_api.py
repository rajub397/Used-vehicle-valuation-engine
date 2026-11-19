from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_invalid_vehicle_rejected():
    bad_payload = {
        "vehicle_type": "Car",
        "brand": "Honda",
        "year": 1985,
        "km_driven": -500,
        "fuel_type": "Rocket",
        "transmission": "Manual",
        "owner_count": 1,
        "engine_cc": 1200
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
