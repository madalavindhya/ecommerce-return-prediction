"""Load the serialized model and create return predictions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import engineer_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.joblib"))


def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Load model metadata and estimator from a joblib artifact."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"Model artifact at {path} has an unsupported format")
    return artifact


def predict_one(model_artifact: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Return the predicted label and probability of a return."""
    frame = pd.DataFrame([values])
    features = engineer_features(frame)
    pipeline = model_artifact["pipeline"]
    prediction = str(pipeline.predict(features)[0])
    classes = list(pipeline.classes_)
    if "Returned" not in classes:
        raise ValueError("The loaded model does not contain the 'Returned' class")
    return_probability = float(
        pipeline.predict_proba(features)[0][classes.index("Returned")]
    )
    return {
        "predicted_return_status": prediction,
        "prediction_probability": return_probability,
        "model_name": model_artifact["model_name"],
    }
