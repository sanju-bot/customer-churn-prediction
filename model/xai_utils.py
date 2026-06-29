from __future__ import annotations

from pathlib import Path

import matplotlib


# Use a headless backend so the plots can be created from the terminal and inside Streamlit.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .runtime import load_shap_module


def _select_positive_class_values(raw_values):
    # SHAP can return either a list of class-specific arrays or one array depending on the version.
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1] if len(raw_values) > 1 else raw_values[0])

    array_values = np.asarray(raw_values)
    if array_values.ndim == 3:
        return array_values[:, :, 1] if array_values.shape[-1] > 1 else array_values[:, :, 0]
    return array_values


def _select_expected_value(expected_value):
    # Binary classifiers sometimes expose a scalar expected value and sometimes a per-class list.
    expected_array = np.asarray(expected_value)
    if expected_array.ndim == 0:
        return float(expected_array)
    if expected_array.size > 1:
        return float(expected_array[1])
    return float(expected_array.reshape(-1)[0])


def get_shap_values(model, data_matrix: np.ndarray):
    # Create a TreeExplainer and return SHAP values for the positive churn class.
    shap = load_shap_module()
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(data_matrix)
    return shap, explainer, _select_positive_class_values(raw_values)


def save_global_summary_plot(model, data_matrix: np.ndarray, feature_names: list[str], output_path: Path, max_features: int = 10):
    # Build one figure with a bar chart and a beeswarm-style scatter so the summary image is useful on its own.
    shap, explainer, shap_values = get_shap_values(model, data_matrix)
    feature_array = np.asarray(data_matrix)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    ordered_indices = np.argsort(mean_abs_shap)[::-1][:max_features]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.0, 1.5]})

    # Left panel: global importance by average absolute SHAP value.
    axes[0].barh(
        np.arange(len(ordered_indices))[::-1],
        mean_abs_shap[ordered_indices][::-1],
        color="#2563eb",
    )
    axes[0].set_yticks(np.arange(len(ordered_indices))[::-1])
    axes[0].set_yticklabels([feature_names[index] for index in ordered_indices][::-1])
    axes[0].set_title("Feature Importance (Mean |SHAP|)")
    axes[0].set_xlabel("Mean absolute SHAP value")

    # Right panel: a simple beeswarm-style scatter that shows how each feature's SHAP values spread out.
    colorbar_reference = None
    for row_index, feature_index in enumerate(ordered_indices[::-1]):
        y_jitter = np.random.default_rng(42 + row_index).normal(loc=row_index, scale=0.08, size=shap_values.shape[0])
        feature_values = feature_array[:, feature_index]
        scatter = axes[1].scatter(
            shap_values[:, feature_index],
            y_jitter,
            c=feature_values,
            cmap="coolwarm",
            alpha=0.7,
            s=18,
            edgecolors="none",
        )
        colorbar_reference = scatter

    axes[1].set_yticks(np.arange(len(ordered_indices)))
    axes[1].set_yticklabels([feature_names[index] for index in ordered_indices[::-1]])
    axes[1].set_title("Beeswarm View of SHAP Values")
    axes[1].set_xlabel("SHAP value")
    axes[1].set_ylabel("Feature")
    if colorbar_reference is not None:
        fig.colorbar(colorbar_reference, ax=axes[1], label="Feature value")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "top_features": [feature_names[index] for index in ordered_indices],
        "mean_abs_shap": mean_abs_shap,
        "shap_values": shap_values,
        "explainer": explainer,
        "shap_module": shap,
    }


def save_waterfall_plot(model, feature_names: list[str], feature_vector: np.ndarray, output_path: Path):
    # Create a SHAP waterfall plot for one customer so the prediction is easy to explain.
    shap, explainer, shap_values = get_shap_values(model, feature_vector.reshape(1, -1))
    base_value = _select_expected_value(explainer.expected_value)
    waterfall_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=feature_vector,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(waterfall_explanation, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return waterfall_explanation


def summarize_waterfall_explanation(waterfall_explanation, top_n: int = 3) -> str:
    # Turn the SHAP output into a short sentence so the user does not have to read the plot alone.
    values = np.asarray(waterfall_explanation.values)
    feature_names = list(waterfall_explanation.feature_names)
    data_values = np.asarray(waterfall_explanation.data)

    ordering = np.argsort(np.abs(values))[::-1][:top_n]
    pieces = []
    for index in ordering:
        direction = "raises" if values[index] > 0 else "lowers"
        pieces.append(f"{feature_names[index]} ({data_values[index]}) {direction} churn")

    if not pieces:
        return "This prediction is based on the combined effect of all customer features."

    return "Key drivers: " + "; ".join(pieces) + "."
