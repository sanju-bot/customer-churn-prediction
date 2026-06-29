from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


# Use a headless backend so charts render correctly inside Streamlit.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


# Make the project root importable when Streamlit launches this file from the app folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.runtime import (  # noqa: E402
    MODEL_DIR,
    build_prediction_input_options,
    csv_text_to_rows,
    get_default_form_values,
    load_artifacts,
    predict_probability,
    predict_rows,
    rows_to_csv_text,
)
from model.xai_utils import save_global_summary_plot, save_waterfall_plot, summarize_waterfall_explanation  # noqa: E402


st.set_page_config(page_title="Customer Churn Prediction + Explainable AI", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        img {
            max-width: 100%;
            height: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_cached_artifacts():
    # Cache the trained model and saved test split so the app stays fast across page switches.
    return load_artifacts()


def risk_card(probability: float) -> None:
    # Show the result with color so high-risk and low-risk predictions are obvious at a glance.
    if probability >= 0.5:
        st.markdown(
            "<div style='padding:1rem;border-radius:0.75rem;background:#fee2e2;color:#991b1b;font-weight:700;'>High Risk of Churn</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='padding:1rem;border-radius:0.75rem;background:#dcfce7;color:#166534;font-weight:700;'>Low Risk of Churn</div>",
            unsafe_allow_html=True,
        )
    st.metric("Churn Probability", f"{probability:.1%}")


def single_prediction_page(model, metadata) -> None:
    # Collect one customer profile so we can generate a prediction and explain it locally.
    st.header("Single Prediction")
    st.write("Enter one customer profile to estimate churn risk and inspect the SHAP explanation.")

    defaults = get_default_form_values(metadata)
    options = build_prediction_input_options(metadata)

    with st.form("single_prediction_form"):
        left_col, right_col = st.columns(2)

        with left_col:
            senior_citizen = st.number_input("SeniorCitizen", min_value=0, max_value=1, value=int(defaults["SeniorCitizen"]))
            tenure = st.number_input("tenure", min_value=0, max_value=72, value=int(defaults["tenure"]))
            monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, value=float(defaults["MonthlyCharges"]), step=0.1)
            total_charges = st.number_input("TotalCharges", min_value=0.0, max_value=10000.0, value=float(defaults["TotalCharges"]), step=1.0)

        with right_col:
            categorical_columns = metadata["categorical_columns"]
            categorical_values = {}
            for index, column_name in enumerate(categorical_columns):
                default_value = str(defaults[column_name])
                categorical_values[column_name] = st.selectbox(
                    column_name,
                    options[column_name],
                    index=options[column_name].index(default_value),
                    key=f"single_{column_name}_{index}",
                )

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        record = {
            "SeniorCitizen": senior_citizen,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            **categorical_values,
        }
        probability, predicted_class, feature_vector = predict_probability(model, metadata, record)
        risk_card(probability)
        st.write("Predicted class:", "Churn" if predicted_class == 1 else "No Churn")

        waterfall_path = MODEL_DIR / "shap_waterfall.png"
        waterfall_explanation = save_waterfall_plot(model, metadata["feature_names"], feature_vector, waterfall_path)
        st.subheader("SHAP Waterfall")
        st.caption(
            "This plot shows which features pushed the prediction toward churn or away from churn. "
            "Positive bars increase churn risk, and negative bars reduce it."
        )
        st.markdown(summarize_waterfall_explanation(waterfall_explanation))
        image_col_left, image_col_center, image_col_right = st.columns([1, 3, 1])
        with image_col_center:
            st.image(str(waterfall_path), width="stretch")


def batch_prediction_page(model, metadata) -> None:
    # Score an uploaded CSV in bulk and let the user download the results.
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV with Telco customer features", type=["csv"])

    if not uploaded_file:
        st.info("Upload a CSV to generate batch predictions.")
        return

    rows = csv_text_to_rows(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
    if not rows:
        st.warning("The uploaded file did not contain any rows.")
        return

    scored_rows = predict_rows(model, metadata, rows)
    st.success(f"Generated predictions for {len(scored_rows)} customers.")
    st.dataframe(scored_rows, use_container_width=True, height=500)

    st.download_button(
        label="Download predictions as CSV",
        data=rows_to_csv_text(scored_rows),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )


def model_performance_page(model, metadata, test_data) -> None:
    # Show holdout performance and the global SHAP summary from the saved test split.
    st.header("Model Performance")
    X_test = np.asarray(test_data["X_test"])
    y_test = np.asarray(test_data["y_test"])
    feature_names = test_data["feature_names"]

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    metric_left, metric_right = st.columns(2)
    metric_left.metric("Accuracy", f"{accuracy:.4f}")
    metric_right.metric("ROC-AUC", f"{roc_auc:.4f}")

    plot_left, plot_right = st.columns(2)
    with plot_left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No", "Yes"])
        ax.set_yticklabels(["No", "Yes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for row_index in range(cm.shape[0]):
            for col_index in range(cm.shape[1]):
                ax.text(col_index, row_index, str(cm[row_index, col_index]), ha="center", va="center", color="black", fontweight="bold")
        fig.colorbar(image, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with plot_right:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#2563eb", label=f"ROC-AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

    summary_path = MODEL_DIR / "shap_summary.png"
    if not summary_path.exists():
        save_global_summary_plot(model, X_test, feature_names, summary_path)

    st.subheader("Global SHAP Feature Importance")
    summary_col_left, summary_col_center, summary_col_right = st.columns([1, 4, 1])
    with summary_col_center:
        st.image(str(summary_path), width="stretch")


def main() -> None:
    # Load the trained model and holdout data once so every page uses the same saved artifacts.
    model, metadata, test_data = load_cached_artifacts()

    st.title("Customer Churn Prediction + Explainable AI")
    st.caption("Single prediction, batch scoring, and model diagnostics in one Streamlit app.")

    page = st.sidebar.radio("Navigate", ["Single Prediction", "Batch Prediction", "Model Performance"])

    if page == "Single Prediction":
        single_prediction_page(model, metadata)
    elif page == "Batch Prediction":
        batch_prediction_page(model, metadata)
    else:
        model_performance_page(model, metadata, test_data)


if __name__ == "__main__":
    main()
