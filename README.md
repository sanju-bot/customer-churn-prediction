# Customer Churn Prediction + Explainable AI

Predict telco customer churn, compare tree-based ML models, and explain individual predictions with SHAP.

## Features

- Explore the IBM Telco Customer Churn dataset from the command line.
- Train and compare XGBoost and Random Forest models.
- Save the best model and preprocessing artifacts for reuse.
- Generate SHAP global and local explanations.
- Run a Streamlit app for single prediction, batch prediction, and model performance.

## Tech Stack

- Python
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Joblib

## Folder Structure

```text
customer-churn-prediction/
├── data/
│   └── telco_churn.csv
├── notebooks/
├── model/
│   ├── train.py
│   ├── churn_model.pkl
│   ├── encoders.pkl
│   ├── test_data.pkl
│   ├── shap_explain.py
│   ├── shap_summary.png
│   └── shap_waterfall.png
├── app/
│   └── streamlit_app.py
├── explore_data.py
├── requirements.txt
└── README.md
```

## How to Run Locally

1. Open the project folder in VS Code.
2. Activate the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Explore the dataset:

   ```powershell
   python explore_data.py
   ```

4. Train the model and save artifacts:

   ```powershell
   python model\train.py
   ```

5. Generate SHAP plots:

   ```powershell
   python model\shap_explain.py
   ```

6. Launch the Streamlit app:

   ```powershell
   streamlit run app\streamlit_app.py
   ```

## Screenshots

Add screenshots here after running the app:

- Single Prediction page
- Batch Prediction page
- Model Performance page

## What I Learned

- How to clean and encode churn data for classification.
- Why ROC-AUC matters for imbalanced classification.
- How to compare multiple tree models fairly.
- How SHAP turns a black-box model into an explainable one.
- How to package a complete ML project with a web UI.

## Notes

- The project is built to work without pandas in this workspace because the Windows policy blocks the pandas binary wheel here.
- The trained model artifacts are stored in the `model/` folder so the app can load them immediately.
# Customer Churn Prediction + Explainable AI

A portfolio project that predicts whether a Telco customer will churn and explains the result with SHAP-based interpretability.

## Features

- Customer churn classification with XGBoost and Random Forest
- Automated preprocessing and model saving
- SHAP-based global and local explainability
- Streamlit dashboard with single prediction, batch prediction, and model performance pages
- CSV-based workflow that works in restricted Windows environments

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Tabular%20ML-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML%20Toolkit-F7931E)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-ff6f61)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b)

## Folder Structure

```text
customer-churn-prediction/
├── data/
│   └── telco_churn.csv
├── notebooks/
├── model/
│   ├── train.py
│   ├── shap_explain.py
│   ├── churn_model.pkl
│   ├── encoders.pkl
│   ├── test_data.pkl
│   ├── shap_summary.png
│   └── shap_waterfall.png
├── app/
│   └── streamlit_app.py
├── churn_utils.py
├── explore_data.py
├── requirements.txt
└── README.md
```

## How to Run Locally

1. Open the project folder in VS Code.
2. Activate the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Make sure the dataset exists at `data/telco_churn.csv`.
5. Run the training phase if you want to regenerate the model artifacts:

   ```powershell
   python model/train.py
   ```

6. Generate SHAP explainability plots:

   ```powershell
   python model/shap_explain.py
   ```

7. Launch the Streamlit app:

   ```powershell
   streamlit run app/streamlit_app.py
   ```

## Screenshots

Add screenshots here after running the app:

- Single Prediction page
- Batch Prediction page
- Model Performance page
- SHAP summary and waterfall plots

## What I Learned / Key Highlights

- How to prepare a real business dataset for churn modeling
- How to compare two tree-based classifiers on an imbalanced problem
- How SHAP helps explain both global feature importance and individual predictions
- How to package a model into a Streamlit app for portfolio presentation

## Notes

- The project uses standard-library CSV handling in the scripts so it can still run on machines where the pandas binary is blocked.
- If you want to version control trained artifacts, remove `*.pkl` from `.gitignore`.
