"""Dataset cleaning and feature engineering for return prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET = "Return_Status"
ID_COLUMN = "Order_ID"
NUMERIC_INPUTS = [
    "Product_Price_INR",
    "Discount_Percent",
    "Expected_Delivery_Days",
    "Actual_Delivery_Days",
    "Customer_Rating_PreOrder",
    "Previous_Returns",
    "Previous_Orders",
]
CATEGORICAL_FEATURES = [
    "Product_Category",
    "Payment_Method",
    "Damaged_Package_Flag",
    "Package_Size",
    "Promotion_Active",
]
NUMERIC_FEATURES = NUMERIC_INPUTS + [
    "delivery_delay",
    "customer_return_frequency",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [ID_COLUMN, *NUMERIC_INPUTS, *CATEGORICAL_FEATURES, TARGET]

_CATEGORY_VALUES = {
    "Product_Category": {
        "beauty": "Beauty",
        "fashion": "Fashion",
        "electronics": "Electronics",
        "home": "Home",
        "sports": "Sports",
    },
    "Payment_Method": {
        "card": "Card",
        "cod": "COD",
        "wallet": "Wallet",
        "upi": "UPI",
    },
    "Damaged_Package_Flag": {"yes": "Yes", "no": "No"},
    "Package_Size": {"small": "Small", "medium": "Medium", "large": "Large"},
    "Promotion_Active": {"yes": "Yes", "no": "No"},
}
_TARGET_VALUES = {
    "returned": "Returned",
    "not returned": "Not Returned",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV using UTF-8/UTF-8-BOM handling and validate its schema."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return frame


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize labels, remove duplicate records, and mark invalid values missing."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    original_rows = len(frame)
    missing_before = frame[REQUIRED_COLUMNS].isna().sum().astype(int).to_dict()
    cleaned = frame[REQUIRED_COLUMNS].copy()
    cleaned = cleaned.drop_duplicates()
    duplicate_rows_removed = original_rows - len(cleaned)

    for column, mapping in _CATEGORY_VALUES.items():
        normalized = cleaned[column].astype("string").str.strip().str.casefold()
        cleaned[column] = normalized.map(mapping)
    target_values = cleaned[TARGET].astype("string").str.strip().str.casefold()
    cleaned[TARGET] = target_values.map(_TARGET_VALUES)
    rows_without_valid_target = int(cleaned[TARGET].isna().sum())
    cleaned = cleaned.dropna(subset=[TARGET]).copy()

    invalid_numeric_counts: dict[str, int] = {}
    for column in NUMERIC_INPUTS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        invalid = cleaned[column].isna()
        if column in {"Product_Price_INR", "Expected_Delivery_Days",
                      "Actual_Delivery_Days", "Customer_Rating_PreOrder",
                      "Previous_Returns", "Previous_Orders"}:
            invalid |= cleaned[column] < 0
        if column == "Discount_Percent":
            invalid |= (cleaned[column] < 0) | (cleaned[column] > 100)
        if column == "Customer_Rating_PreOrder":
            invalid |= cleaned[column] > 5
        if column in {"Previous_Returns", "Previous_Orders"}:
            invalid |= cleaned[column] % 1 != 0
        invalid_numeric_counts[column] = int(invalid.sum())
        cleaned.loc[invalid, column] = np.nan

    impossible_returns = (
        cleaned["Previous_Returns"].notna()
        & cleaned["Previous_Orders"].notna()
        & (cleaned["Previous_Returns"] > cleaned["Previous_Orders"])
    )
    invalid_numeric_counts["Previous_Returns"] += int(impossible_returns.sum())
    cleaned.loc[impossible_returns, "Previous_Returns"] = np.nan

    info: dict[str, Any] = {
        "source_rows": original_rows,
        "rows_after_deduplication": int(original_rows - duplicate_rows_removed),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "rows_without_valid_target_removed": rows_without_valid_target,
        "rows_after_target_validation": int(len(cleaned)),
        "missing_values_before_cleaning": missing_before,
        "invalid_numeric_values_marked_missing": invalid_numeric_counts,
        "missing_values_after_cleaning": cleaned[REQUIRED_COLUMNS]
            .isna().sum().astype(int).to_dict(),
        "target_distribution": cleaned[TARGET].value_counts().to_dict(),
        "excluded_from_model_features": [ID_COLUMN],
    }
    return cleaned.reset_index(drop=True), info


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create delivery delay and prior-return frequency without using identifiers."""
    features = frame.drop(columns=[ID_COLUMN, TARGET], errors="ignore").copy()
    for column in NUMERIC_INPUTS:
        if column not in features:
            raise ValueError(f"Missing prediction feature: {column}")
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features["delivery_delay"] = (
        features["Actual_Delivery_Days"] - features["Expected_Delivery_Days"]
    )
    previous_orders = features["Previous_Orders"].replace(0, np.nan)
    features["customer_return_frequency"] = (
        features["Previous_Returns"] / previous_orders
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features[FEATURE_COLUMNS]


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write JSON data with stable, human-readable formatting."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
