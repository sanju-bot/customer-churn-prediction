from __future__ import annotations

import csv
import io
import sys
import types
from pathlib import Path
from typing import Any

import joblib
import numpy as np


# Keep the main project paths in one place so every script points to the same artifacts.
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
ENCODER_PATH = MODEL_DIR / "encoders.pkl"
TEST_DATA_PATH = MODEL_DIR / "test_data.pkl"
SHAP_SUMMARY_PATH = MODEL_DIR / "shap_summary.png"
SHAP_WATERFALL_PATH = MODEL_DIR / "shap_waterfall.png"


def build_pandas_stub() -> types.ModuleType:
    # Create a lightweight in-memory pandas replacement so SHAP can import in this restricted environment.
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.__version__ = "0.0"
    pandas_stub.Series = type("Series", (), {})
    pandas_stub.DataFrame = type("DataFrame", (), {})
    pandas_stub.Index = type("Index", (), {})
    pandas_stub.MultiIndex = type("MultiIndex", (), {})
    pandas_stub.RangeIndex = type("RangeIndex", (), {})
    pandas_stub.CategoricalDtype = type("CategoricalDtype", (), {})
    pandas_stub.Categorical = type("Categorical", (), {})
    pandas_stub.Timestamp = type("Timestamp", (), {})
    pandas_stub.Timedelta = type("Timedelta", (), {})
    pandas_stub.NA = object()
    pandas_stub.api = types.SimpleNamespace(
        types=types.SimpleNamespace(
            is_object_dtype=lambda *_args, **_kwargs: False,
            is_string_dtype=lambda *_args, **_kwargs: False,
            is_numeric_dtype=lambda *_args, **_kwargs: False,
        )
    )
    pandas_stub.isna = lambda _value: False
    pandas_stub.notna = lambda _value: True
    pandas_stub.concat = lambda *_args, **_kwargs: None
    pandas_stub.unique = lambda values: list(dict.fromkeys(values))
    pandas_stub.to_numeric = lambda values, **_kwargs: values
    pandas_stub.array = lambda values, **_kwargs: values
    pandas_stub.factorize = lambda values, **_kwargs: (list(range(len(values))), [])
    pandas_stub.get_dummies = lambda *_args, **_kwargs: None
    pandas_stub.read_csv = lambda *_args, **_kwargs: None
    return pandas_stub


def import_shap():
    # Try the normal import first so we only use the stub if the machine blocks pandas from loading.
    try:
        import shap  # type: ignore

        return shap
    except ImportError as original_error:
        error_text = str(original_error)
        if "pandas" not in error_text and "DLL load failed" not in error_text:
            raise

        # Register the stub and retry so SHAP can be used with numpy arrays and tree-based models.
        sys.modules["pandas"] = build_pandas_stub()
        import shap  # type: ignore

        return shap


def load_model_bundle() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    # Load the trained model and preprocessing metadata that the training phase already saved.
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(ENCODER_PATH)
    test_data = joblib.load(TEST_DATA_PATH)
    return model, metadata, test_data


def parse_csv_rows(file_bytes: bytes) -> list[dict[str, str]]:
    # Decode uploaded bytes and parse them with the standard library so we do not depend on pandas.
    text = file_bytes.decode("utf-8-sig")
    with io.StringIO(text) as buffer:
        return list(csv.DictReader(buffer))


def clean_numeric(value: Any, fallback: float = 0.0) -> float:
    # Convert a widget value or CSV cell into a float, falling back to a safe default when needed.
    if value is None:
        return float(fallback)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return float(fallback)
        try:
            return float(stripped)
        except ValueError:
            return float(fallback)

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def encode_record(record: dict[str, Any], metadata: dict[str, Any]) -> np.ndarray:
    # Convert one customer row into the exact numeric feature order the model was trained on.
    numeric_columns = metadata["numeric_columns"]
    categorical_columns = metadata["categorical_columns"]
    label_encoders = metadata["label_encoders"]
    total_charges_median = float(metadata.get("total_charges_median", 0.0))

    feature_values: list[float] = []

    for column_name in numeric_columns:
        if column_name == "TotalCharges":
            feature_values.append(clean_numeric(record.get(column_name), fallback=total_charges_median))
        else:
            feature_values.append(clean_numeric(record.get(column_name), fallback=0.0))

    for column_name in categorical_columns:
        encoder = label_encoders[column_name]
        allowed_values = list(encoder.classes_)
        raw_value = str(record.get(column_name, "")).strip() or allowed_values[0]
        if raw_value not in allowed_values:
            raw_value = allowed_values[0]
        encoded_value = int(encoder.transform([raw_value])[0])
        feature_values.append(float(encoded_value))

    return np.array(feature_values, dtype=float).reshape(1, -1)


def build_default_customer(metadata: dict[str, Any]) -> dict[str, Any]:
    # Provide sensible starting values so the Streamlit form opens with a complete example customer.
    label_encoders = metadata["label_encoders"]

    defaults = {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": float(metadata.get("total_charges_median", 0.0)),
    }

    for column_name, encoder in label_encoders.items():
        defaults[column_name] = encoder.classes_[0]

    return defaults


def dicts_to_html_table(rows: list[dict[str, Any]]) -> str:
    # Render prediction rows as simple HTML so the app can display a table without pandas.
    if not rows:
        return "<p>No rows to display.</p>"

    columns = list(rows[0].keys())
    header_html = "".join(f"<th>{column}</th>" for column in columns)
    body_html = ""
    for row in rows:
        row_html = "".join(f"<td>{row.get(column, '')}</td>" for column in columns)
        body_html += f"<tr>{row_html}</tr>"

    return f"<table style='width:100%; border-collapse:collapse;' border='1'><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def serialize_rows_as_csv(rows: list[dict[str, Any]]) -> str:
    # Turn prediction rows into CSV text so users can download the batch results.
    if not rows:
        return ""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
