from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# Keep all project paths in one place so the script stays easy to reuse from VS Code or the terminal.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
ENCODER_PATH = MODEL_DIR / "encoders.pkl"
TEST_DATA_PATH = MODEL_DIR / "test_data.pkl"


# These are the columns in the Telco dataset after dropping customerID and separating out the target.
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


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    # Read the CSV into dictionaries so each row can be cleaned column by column.
    with csv_path.open(newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def safe_float(value: str) -> float:
    # Convert numeric text to float while tolerating blank strings and stray whitespace.
    stripped_value = value.strip()
    if stripped_value == "":
        return math.nan
    return float(stripped_value)


def prepare_matrix(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, LabelEncoder], float]:
    # Remove customerID because it is just an identifier and does not help the model learn churn behavior.
    feature_rows = []
    target_values = []

    # Split the target from the feature columns before encoding so we can train and evaluate cleanly.
    for row in rows:
        target_values.append(1 if row[TARGET_COLUMN].strip().lower() == "yes" else 0)
        feature_rows.append({column: row[column] for column in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS})

    # Build the numeric matrix first because these columns are already meaningful on their original scale.
    numeric_matrix = np.zeros((len(feature_rows), len(NUMERIC_COLUMNS)), dtype=float)
    for row_index, row in enumerate(feature_rows):
        for column_index, column_name in enumerate(NUMERIC_COLUMNS):
            numeric_matrix[row_index, column_index] = safe_float(row[column_name])

    # Impute missing TotalCharges values with the median so blank entries do not break the training step.
    total_charges_index = NUMERIC_COLUMNS.index("TotalCharges")
    total_charges_column = numeric_matrix[:, total_charges_index]
    total_charges_median = float(np.nanmedian(total_charges_column))
    total_charges_column = np.where(np.isnan(total_charges_column), total_charges_median, total_charges_column)
    numeric_matrix[:, total_charges_index] = total_charges_column

    # Encode each categorical column so the tree models can use them as numeric inputs.
    encoded_columns = []
    encoders: dict[str, LabelEncoder] = {}
    for column_name in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        column_values = [row[column_name].strip() if row[column_name].strip() else "Missing" for row in feature_rows]
        encoded_values = encoder.fit_transform(column_values)
        encoders[column_name] = encoder
        encoded_columns.append(encoded_values.reshape(-1, 1).astype(float))

    # Combine the numeric and encoded categorical features into one matrix for model training.
    if encoded_columns:
        feature_matrix = np.hstack([numeric_matrix] + encoded_columns)
    else:
        feature_matrix = numeric_matrix

    # Build feature names in the same order as the matrix so saved artifacts stay consistent later.
    feature_names = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    # Return the cleaned features, targets, feature names, and encoders for downstream training and inference.
    return feature_matrix, np.array(target_values, dtype=int), feature_names, encoders, total_charges_median


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    # Generate predictions so we can report both classification quality and ranking quality.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, predictions))

    # Return the key metrics so we can compare the candidate models and keep the best one.
    return {"accuracy": accuracy, "roc_auc": roc_auc}


def main() -> None:
    # Confirm the dataset exists before doing any work so the failure mode is easy to understand.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Put the Telco CSV in data/telco_churn.csv first.")

    # Load the rows into memory so we can clean them without relying on pandas in this environment.
    rows = load_rows(DATA_PATH)

    if not rows:
        raise ValueError(f"{DATA_PATH} is empty.")

    # Build a fully numeric feature matrix and keep the encoders needed for later inference.
    X, y, feature_names, encoders, total_charges_median = prepare_matrix(rows)

    # Split the data once so both candidate models are trained and evaluated on the exact same holdout set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train XGBoost because it usually performs very well on tabular customer churn problems.
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        n_jobs=4,
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost results:")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
    print()

    # Train Random Forest as a second baseline because it is robust and easy to interpret.
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=4,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)
    print("Random Forest results:")
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print()

    # Choose the better model by ROC-AUC first because the churn problem is imbalanced.
    if rf_metrics["roc_auc"] > xgb_metrics["roc_auc"]:
        best_model = rf_model
        best_model_name = "RandomForestClassifier"
        best_metrics = rf_metrics
    else:
        best_model = xgb_model
        best_model_name = "XGBClassifier"
        best_metrics = xgb_metrics

    # Save the winning model so the Streamlit app and SHAP analysis can reuse it later.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    # Save the encoders and preprocessing metadata so future inference uses the same feature transformations.
    joblib.dump(
        {
            "label_encoders": encoders,
            "feature_names": feature_names,
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "total_charges_median": total_charges_median,
            "best_model_name": best_model_name,
            "best_metrics": best_metrics,
        },
        ENCODER_PATH,
    )

    # Save the test split so the SHAP phase can explain the same holdout samples used for evaluation.
    joblib.dump(
        {
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_names,
        },
        TEST_DATA_PATH,
    )

    # Print the final model choice so the user can see which approach won and why.
    print(f"Best model saved: {best_model_name}")
    print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()