from __future__ import annotations

import sys
from pathlib import Path


# Add the project root to sys.path so the script works when run as python model/shap_explain.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.runtime import MODEL_DIR, load_artifacts  # noqa: E402
from model.xai_utils import save_global_summary_plot, save_waterfall_plot  # noqa: E402


def main() -> None:
    # Load the trained model plus the holdout test data saved during Phase 2.
    model, metadata, test_data = load_artifacts()
    X_test = test_data["X_test"]
    feature_names = test_data["feature_names"]

    summary_path = MODEL_DIR / "shap_summary.png"
    waterfall_path = MODEL_DIR / "shap_waterfall.png"

    # Generate a global summary image so we can explain which features drive churn overall.
    summary_info = save_global_summary_plot(model, X_test, feature_names, summary_path)

    # Generate a waterfall explanation for the first test customer so one prediction is transparent.
    save_waterfall_plot(model, feature_names, X_test[0], waterfall_path)

    # Print the strongest drivers so the user can sanity-check the SHAP output from the terminal.
    print("Top 5 important features:")
    for feature_name in summary_info["top_features"][:5]:
        print(feature_name)
    print(f"Saved summary plot to: {summary_path}")
    print(f"Saved waterfall plot to: {waterfall_path}")


if __name__ == "__main__":
    main()
