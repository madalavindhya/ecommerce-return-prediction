"""Train, evaluate, log, and serialize e-commerce return classifiers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    clean_dataset,
    engineer_features,
    load_dataset,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv(
    "DATA_PATH",
    str(ROOT / "data" / "Section7_Q4_Ecommerce_Product_Return_CICD.csv"),
))
MODEL_PATH = ROOT / "models" / "best_model.joblib"
REPORT_PATH = ROOT / "reports" / "evaluation_results.json"
EXPERIMENT_NAME = "Ecommerce_Product_Return_Prediction"
RANDOM_STATE = 42


def _build_pipeline(estimator: Any) -> Pipeline:
    numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformer = ColumnTransformer([
        ("numerical", numerical, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", transformer), ("classifier", estimator)])


def _metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x_test)
    labels = ["Not Returned", "Returned"]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, pos_label="Returned", zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, pos_label="Returned", zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions, pos_label="Returned", zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist(),
        "confusion_matrix_labels": labels,
        "classification_report": classification_report(
            y_test, predictions, labels=labels, zero_division=0, output_dict=True
        ),
    }


def train() -> dict[str, Any]:
    """Run reproducible model training and return the persisted evaluation report."""
    raw = load_dataset(DATA_PATH)
    cleaned, preprocessing_info = clean_dataset(raw)
    features = engineer_features(cleaned)

    labels = cleaned[TARGET]
    stratify = labels if labels.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    candidates: list[tuple[str, Any, dict[str, Any]]] = [
        (
            "LogisticRegression",
            LogisticRegression(class_weight="balanced", max_iter=1500, random_state=RANDOM_STATE),
            {"class_weight": "balanced", "max_iter": 1500},
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "class_weight": "balanced_subsample",
            },
        ),
    ]

    default_tracking_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    runs: list[dict[str, Any]] = []
    for model_name, estimator, params in candidates:
        pipeline = _build_pipeline(estimator)
        with mlflow.start_run(run_name=model_name) as active_run:
            pipeline.fit(x_train, y_train)
            metrics = _metrics(pipeline, x_test, y_test)
            mlflow.log_params({
                **params,
                "test_size": 0.2,
                "random_state": RANDOM_STATE,
                "stratified_split": stratify is not None,
                "class_imbalance_handling": params["class_weight"],
            })
            mlflow.log_metrics({
                key: metrics[key]
                for key in ("accuracy", "precision", "recall", "f1_score")
            })
            mlflow.set_tags({
                "training_dataset": DATA_PATH.name,
                "training_rows": len(cleaned),
                "training_columns": len(raw.columns),
                "feature_count": len(FEATURE_COLUMNS),
                "target": TARGET,
            })
            signature = infer_signature(x_train, pipeline.predict(x_train))
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=x_train.head(3),
            )
            runs.append({
                "model_name": model_name,
                "params": params,
                "metrics": metrics,
                "run_id": active_run.info.run_id,
                "pipeline": pipeline,
            })

    best = max(
        runs,
        key=lambda item: (
            item["metrics"]["f1_score"],
            item["metrics"]["recall"],
            item["metrics"]["accuracy"],
        ),
    )
    fitted_preprocessor = best["pipeline"].named_steps["preprocessor"]
    encoded_features = fitted_preprocessor.transform(features)
    if hasattr(encoded_features, "toarray"):
        encoded_features = encoded_features.toarray()
    processed = pd.DataFrame(
        encoded_features,
        columns=fitted_preprocessor.get_feature_names_out(),
    )
    processed[TARGET] = labels.reset_index(drop=True)
    preprocessing_info["model_preprocessing"] = {
        "numeric_imputation": "training-split median",
        "numeric_scaling": "StandardScaler",
        "categorical_imputation": "training-split most frequent",
        "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        "fitted_without_test_data": True,
        "encoded_feature_count": len(fitted_preprocessor.get_feature_names_out()),
        "encoded_feature_columns": list(fitted_preprocessor.get_feature_names_out()),
    }
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_dir / "processed_dataset.csv", index=False)
    write_json(preprocessing_info, processed_dir / "preprocessing_info.json")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": best["pipeline"],
        "model_name": best["model_name"],
        "target_classes": list(best["pipeline"].classes_),
    }, MODEL_PATH)

    registry_name = "EcommerceProductReturnBestModel"
    model_uri = f"runs:/{best['run_id']}/model"
    registered_model = mlflow.register_model(model_uri, registry_name)
    result: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "tracking_uri": tracking_uri,
        "training_dataset": {
            "file": DATA_PATH.name,
            "original_shape": list(raw.shape),
            "processed_shape": list(processed.shape),
            "columns": list(raw.columns),
            "target_distribution": preprocessing_info["target_distribution"],
        },
        "preprocessing": preprocessing_info,
        "feature_engineering": {
            "features": ["delivery_delay", "customer_return_frequency"],
            "discount_feature": "Discount_Percent retained and range-validated [0, 100]",
            "excluded_features": ["Order_ID"],
            "zero_previous_orders_frequency": 0.0,
        },
        "split": {
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "stratified": stratify is not None,
            "random_state": RANDOM_STATE,
        },
        "model_comparison": [
            {
                "model_name": item["model_name"],
                "params": item["params"],
                "metrics": item["metrics"],
                "mlflow_run_id": item["run_id"],
            }
            for item in runs
        ],
        "best_model": {
            "model_name": best["model_name"],
            "selection_criteria": "Highest Returned-class F1-score, then recall, then accuracy",
            "metrics": best["metrics"],
            "mlflow_run_id": best["run_id"],
            "registered_model_name": registry_name,
            "registered_model_version": registered_model.version,
            "artifact_path": str(MODEL_PATH.relative_to(ROOT)),
        },
    }
    write_json(result, REPORT_PATH)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_artifact(str(REPORT_PATH), artifact_path="reports")

    print(f"Dataset shape: {raw.shape}")
    print(f"Columns: {list(raw.columns)}")
    print("Missing values before cleaning:")
    print(raw.isna().sum().to_string())
    print(f"Preprocessing summary: {json.dumps(preprocessing_info, indent=2)}")
    print("Engineered features (first 5 rows):")
    print(features[["delivery_delay", "customer_return_frequency", "Discount_Percent"]]
          .head().to_string(index=False))
    print("Model comparison:")
    print(pd.DataFrame([
        {
            "model": item["model_name"],
            **{metric: item["metrics"][metric]
               for metric in ("accuracy", "precision", "recall", "f1_score")},
        }
        for item in runs
    ]).to_string(index=False))
    print(f"Best model: {best['model_name']}")
    print(f"Confusion matrix (rows actual, columns predicted; {best['metrics']['confusion_matrix_labels']}):")
    print(best["metrics"]["confusion_matrix"])
    print("Best model classification report:")
    print(pd.DataFrame(best["metrics"]["classification_report"]).transpose().to_string())
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Registered model: {registry_name} version {registered_model.version}")
    print(f"Evaluation report: {REPORT_PATH}")
    print(f"Serialized model: {MODEL_PATH}")
    return result


if __name__ == "__main__":
    train()
