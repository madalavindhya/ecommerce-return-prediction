import json
from pathlib import Path

import pandas as pd

from src.predict import load_model, predict_one
from src.preprocessing import clean_dataset, engineer_features, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Section7_Q4_Ecommerce_Product_Return_CICD.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
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


def test_dataset_load_and_preprocessing():
    raw = load_dataset(DATA_PATH)
    cleaned, info = clean_dataset(raw)
    features = engineer_features(cleaned)
    assert raw.shape == (1200, 14)
    assert len(cleaned) == 1200
    assert info["duplicate_rows_removed"] == 0
    assert "Order_ID" not in features.columns
    assert {"delivery_delay", "customer_return_frequency"} <= set(features.columns)
    assert features["Discount_Percent"].dropna().between(0, 100).all()
    processed = pd.read_csv(ROOT / "data" / "processed" / "processed_dataset.csv")
    assert "Order_ID" not in processed.columns
    assert processed.isna().sum().sum() == 0
    assert "Return_Status" in processed.columns
    assert any(column.endswith("Discount_Percent") for column in processed.columns)


def test_engineered_features_handle_zero_orders():
    values = {**SAMPLE, "Previous_Returns": 0, "Previous_Orders": 0}
    engineered = engineer_features(pd.DataFrame([values]))
    assert engineered.loc[0, "delivery_delay"] == 2
    assert engineered.loc[0, "customer_return_frequency"] == 0


def test_serialized_model_predicts_valid_result():
    model = load_model(MODEL_PATH)
    result = predict_one(model, SAMPLE)
    assert result["predicted_return_status"] in {"Returned", "Not Returned"}
    assert 0 <= result["prediction_probability"] <= 1
    assert result["model_name"]
    report = json.loads((ROOT / "reports" / "evaluation_results.json").read_text())
    assert len(report["model_comparison"]) >= 2
