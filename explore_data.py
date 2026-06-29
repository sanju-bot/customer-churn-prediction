from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path


# Point directly at the dataset file in the data folder so the script matches the project layout.
DATA_PATH = Path("data") / "telco_churn.csv"


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    # Read the CSV into memory so we can inspect its structure and produce summary statistics.
    with csv_path.open(newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def infer_dtype(values: list[str]) -> str:
    # Treat blanks as missing values and infer a simple dtype from the non-empty cells.
    cleaned_values = [value.strip() for value in values if value and value.strip() != ""]

    if not cleaned_values:
        return "object"

    def is_int(value: str) -> bool:
        try:
            int(value)
            return True
        except ValueError:
            return False

    def is_float(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False

    if all(is_int(value) for value in cleaned_values):
        return "int64"

    if all(is_float(value) for value in cleaned_values):
        return "float64"

    return "object"


def describe_numeric(values: list[float]) -> dict[str, float]:
    # Compute basic descriptive statistics for numeric columns so we can spot scale and spread.
    ordered = sorted(values)
    count = len(ordered)
    mean_value = sum(ordered) / count if count else math.nan
    variance = sum((value - mean_value) ** 2 for value in ordered) / count if count else math.nan
    std_value = math.sqrt(variance) if count else math.nan

    def percentile(percent: float) -> float:
        if not ordered:
            return math.nan
        index = (len(ordered) - 1) * percent
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[int(index)]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": float(count),
        "mean": mean_value,
        "std": std_value,
        "min": ordered[0] if ordered else math.nan,
        "25%": percentile(0.25),
        "50%": percentile(0.50),
        "75%": percentile(0.75),
        "max": ordered[-1] if ordered else math.nan,
    }


def main() -> None:
    # Stop early with a clear message if the CSV is missing, because the rest of the script depends on it.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Make sure the Telco dataset is saved there as telco_churn.csv."
        )

    # Load the CSV rows so we can inspect the structure, types, missing values, and target balance.
    rows = load_rows(DATA_PATH)

    if not rows:
        raise ValueError(f"{DATA_PATH} is empty.")

    # Pull the column names from the CSV header so we can report the feature layout.
    columns = list(rows[0].keys())

    # Print the dataset shape first so we know how many rows and columns we are working with.
    print("Shape:")
    print((len(rows), len(columns)))
    print()

    # Print the column list so we can confirm the available features and target column names.
    print("Column names:")
    print(columns)
    print()

    # Infer dtypes column by column so we can spot fields that look numeric versus categorical.
    print("Data types:")
    dtypes = {}
    column_values = {column: [row[column] for row in rows] for column in columns}
    for column in columns:
        dtypes[column] = infer_dtype(column_values[column])
    for column, dtype_name in dtypes.items():
        print(f"{column}    {dtype_name}")
    print()

    # Count missing values per column so we know which fields may need cleaning or imputation.
    print("Missing values per column:")
    for column in columns:
        missing_count = sum(1 for value in column_values[column] if value is None or value.strip() == "")
        print(f"{column}    {missing_count}")
    print()

    # Print the Churn distribution because the target balance affects model choice and evaluation.
    if "Churn" in columns:
        print("Target value counts for 'Churn':")
        churn_counts = Counter(value if value.strip() != "" else "<missing>" for value in column_values["Churn"])
        for label, count in churn_counts.items():
            print(f"{label}    {count}")
        print()
    else:
        print("Column 'Churn' was not found in the dataset.")
        print()

    # Print summary statistics so we can quickly understand numeric ranges and categorical coverage.
    print("Describe statistics:")
    for column in columns:
        values = column_values[column]
        numeric_values = []
        for value in values:
            stripped = value.strip() if value is not None else ""
            if stripped == "":
                continue
            try:
                numeric_values.append(float(stripped))
            except ValueError:
                numeric_values = []
                break

        if numeric_values:
            stats = describe_numeric(numeric_values)
            print(column)
            for key, stat_value in stats.items():
                print(f"  {key}    {stat_value}")
            print()
        else:
            non_empty = [value for value in values if value.strip() != ""]
            unique_values = list(dict.fromkeys(non_empty))
            top_value = Counter(non_empty).most_common(1)[0] if non_empty else (None, 0)
            print(column)
            print(f"  count    {len(non_empty)}")
            print(f"  unique   {len(unique_values)}")
            print(f"  top      {top_value[0]}")
            print(f"  freq     {top_value[1]}")
            print()


# Keep the script runnable from the command line without executing on import.
if __name__ == "__main__":
    main()