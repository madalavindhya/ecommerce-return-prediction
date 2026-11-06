from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
SAMPLE = {
    "Product_Category": "Fashion",
    "Product_Price_INR": 2500,
    "Discount_Percent": 20,
    "Expected_Delivery_Days": 5,
    "Actual_Delivery_Days": 7,
    "Customer_Rating_PreOrder": 3.5,
    "Previous_Returns": 2,
    "Previous_Orders": 10,
    "Payment_Method": "Card",
    "Damaged_Package_Flag": "No",
    "Package_Size": "Medium",
    "Promotion_Active": "Yes",
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_response_shape():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "predicted_return_status",
        "prediction_probability",
        "model_name",
    }
    assert body["predicted_return_status"] in {"Returned", "Not Returned"}
    assert 0 <= body["prediction_probability"] <= 1


def test_invalid_api_inputs_are_rejected():
    invalid = {**SAMPLE, "Discount_Percent": 120}
    assert client.post("/predict", json=invalid).status_code == 422
    assert client.post("/predict", json={**SAMPLE, "Unexpected": "field"}).status_code == 422
