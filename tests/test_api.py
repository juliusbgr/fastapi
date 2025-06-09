from fastapi.testclient import TestClient
from app.main import app
import os

os.environ["API_TOKEN"] = "test-secret"

client = TestClient(app)

valid_payload = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

def test_predict_success():
    response = client.post(
        "/predict",
        json=valid_payload,
        headers={"X-API-Token": "test-secret"}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_invalid_token():
    response = client.post(
        "/predict",
        json=valid_payload,
        headers={"X-API-Token": "wrong-token"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"
