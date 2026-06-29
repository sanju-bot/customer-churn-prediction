from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np


# Keep project paths in one place so training, SHAP, and the app all use the same artifacts.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
ENCODER_PATH = MODEL_DIR / "encoders.pkl"
TEST_DATA_PATH = MODEL_DIR / "test_data.pkl"


# These columns match the exact feature order used in training after customerID is dropped.
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET_COLUMN = "Churn"


def load_artifacts() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    # Load the saved model, preprocessing metadata, and test split so other phases can reuse them.
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(ENCODER_PATH)
    test_data = joblib.load(TEST_DATA_PATH)
    return model, metadata, test_data


def load_model_only() -> Any:
    # Load just the trained model when the caller does not need preprocessing details.
    return joblib.load(MODEL_PATH)


def safe_float(value: Any) -> float:
    # Convert form values or CSV text to float while tolerating blanks and whitespace.
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    stripped_value = str(value).strip()
    if stripped_value == "":
        return math.nan
    return float(stripped_value)


def normalize_categorical_value(encoder, value: Any) -> str:
    # Keep the value inside the encoder vocabulary so prediction never fails on a dropdown choice.
    cleaned_value = "" if value is None else str(value).strip()
    if cleaned_value == "":
        return str(encoder.classes_[0])
    if cleaned_value in encoder.classes_:
        return cleaned_value
    return str(encoder.classes_[0])


def build_feature_vector(row: dict[str, Any], metadata: dict[str, Any]) -> np.ndarray:
    # Rebuild the exact numeric feature vector that the model saw during training.
    encoders = metadata["label_encoders"]
    total_charges_median = float(metadata["total_charges_median"])

    numeric_values: list[float] = []
    for column_name in NUMERIC_COLUMNS:
        numeric_value = safe_float(row.get(column_name))
        if column_name == "TotalCharges" and math.isnan(numeric_value):
            numeric_value = total_charges_median
        if column_name == "SeniorCitizen" and not math.isnan(numeric_value):
            numeric_value = int(numeric_value)
        numeric_values.append(numeric_value)

    categorical_values: list[float] = []
    for column_name in CATEGORICAL_COLUMNS:
        encoder = encoders[column_name]
        cleaned_value = normalize_categorical_value(encoder, row.get(column_name))
        categorical_values.append(float(encoder.transform([cleaned_value])[0]))

    return np.asarray(numeric_values + categorical_values, dtype=float)


def predict_probability(model: Any, metadata: dict[str, Any], row: dict[str, Any]) -> tuple[float, int, np.ndarray]:
    # Run a single customer record through the trained model and return both probability and class.
    feature_vector = build_feature_vector(row, metadata)
    probability = float(model.predict_proba(feature_vector.reshape(1, -1))[0, 1])
    predicted_class = int(probability >= 0.5)
    return probability, predicted_class, feature_vector


def predict_rows(model: Any, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Score a batch of uploaded customers and attach prediction outputs to each row.
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        probability, predicted_class, _ = predict_probability(model, metadata, row)
        result_row = dict(row)
        result_row["Churn_Probability"] = round(probability, 4)
        result_row["Predicted_Churn"] = "Yes" if predicted_class == 1 else "No"
        output_rows.append(result_row)
    return output_rows


def get_default_form_values(metadata: dict[str, Any]) -> dict[str, Any]:
    # Provide sensible defaults so the Streamlit form opens with realistic customer values.
    encoders = metadata["label_encoders"]
    defaults: dict[str, Any] = {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 65.0,
        "TotalCharges": float(metadata["total_charges_median"]),
    }
    for column_name in CATEGORICAL_COLUMNS:
        defaults[column_name] = str(encoders[column_name].classes_[0])
    return defaults


def build_prediction_input_options(metadata: dict[str, Any]) -> dict[str, list[str]]:
    # Convert each fitted label encoder into a list of choices for select boxes.
    encoders = metadata["label_encoders"]
    return {column_name: [str(value) for value in encoders[column_name].classes_] for column_name in CATEGORICAL_COLUMNS}


def csv_text_to_rows(csv_text: str) -> list[dict[str, str]]:
    # Parse an uploaded CSV into row dictionaries using the standard library only.
    reader = csv.DictReader(io.StringIO(csv_text))
    return list(reader)


def rows_to_csv_text(rows: list[dict[str, Any]]) -> str:
    # Serialize prediction results back to CSV so users can download the scored batch.
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def install_pandas_stub() -> None:
    # SHAP imports pandas at module load time, and this environment blocks the compiled pandas wheel.
    # A small stub is enough for SHAP's TreeExplainer paths that operate on NumPy arrays.
    import sys
    import types

    if "pandas" in sys.modules:
        return

    pandas_stub = types.ModuleType("pandas")
    pandas_stub.__version__ = "0.0"
    pandas_stub.Series = type("Series", (), {})
    pandas_stub.DataFrame = type("DataFrame", (), {})
    pandas_stub.Index = type("Index", (), {})
    pandas_stub.MultiIndex = type("MultiIndex", (), {})
    pandas_stub.CategoricalDtype = type("CategoricalDtype", (), {})
    pandas_stub.RangeIndex = type("RangeIndex", (), {})
    pandas_stub.Categorical = type("Categorical", (), {})
    pandas_stub.api = types.SimpleNamespace(types=types.SimpleNamespace())
    pandas_stub.isna = lambda value: value is None or (isinstance(value, float) and math.isnan(value))
    pandas_stub.notna = lambda value: not pandas_stub.isna(value)
    pandas_stub.concat = lambda *args, **kwargs: None
    pandas_stub.unique = lambda values: list(dict.fromkeys(values))
    pandas_stub.to_numeric = lambda values, **kwargs: values
    pandas_stub.read_csv = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pandas stub cannot read CSV"))

    sys.modules["pandas"] = pandas_stub


def load_shap_module():
    # Import SHAP only after the stub is installed so the module can load in this environment.
    import importlib

    install_pandas_stub()
    return importlib.import_module("shap")
