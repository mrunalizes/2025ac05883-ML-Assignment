"""
app.py
------
Streamlit app for the Heart Disease Risk (2026) classification project.

Loads the 5 pre-trained models from the model/ folder. If a file isn't
found locally (e.g. deployment checkout issue), it's downloaded directly
from this GitHub repo's raw file URLs as a fallback.

Expected project structure:
    project-folder/
    │-- app.py
    │-- requirements.txt
    │-- README.md
    │-- test_data.csv
    │-- heart_disease_risk_2026.csv
    │-- model/
    │   │-- train_models.py
    │   │-- logistic_regression.joblib
    │   │-- decision_tree.joblib
    │   │-- knn.joblib
    │   │-- naive_bayes.joblib
    │   │-- random_forest_ensemble.joblib
    │   │-- metrics_comparison.csv

IMPORTANT: update GITHUB_USER / GITHUB_REPO / GITHUB_BRANCH below to match
your actual repo.
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Heart Disease Risk Classifier", layout="wide")

# ---- update these three if your repo details differ ----
GITHUB_USER = "mrunalizes"
GITHUB_REPO = "2025ac05883-ml-assignment"
GITHUB_BRANCH = "main"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}"

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}
TARGET = "has_heart_disease"


def ensure_file_present(relative_path: str) -> str:
    """Return a local path to `relative_path` (e.g. 'model/knn.joblib' or
    'test_data.csv'), downloading it from GitHub raw if not found locally."""
    if os.path.exists(relative_path):
        return relative_path

    os.makedirs(os.path.dirname(relative_path), exist_ok=True) if os.path.dirname(relative_path) else None
    url = f"{RAW_BASE_URL}/{relative_path}"
    with st.spinner(f"Fetching {relative_path} from GitHub..."):
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            raise FileNotFoundError(
                f"Could not download '{relative_path}' from {url} "
                f"(status {response.status_code}). Check GITHUB_USER/GITHUB_REPO/"
                f"GITHUB_BRANCH at the top of app.py and confirm the file exists "
                f"in the repo at that exact path."
            )
        with open(relative_path, "wb") as f:
            f.write(response.content)
    return relative_path


@st.cache_resource
def load_model(filename: str):
    local_path = ensure_file_present(f"{MODEL_DIR}/{filename}")
    return joblib.load(local_path)


@st.cache_data
def load_comparison_table():
    local_path = ensure_file_present(f"{MODEL_DIR}/metrics_comparison.csv")
    return pd.read_csv(local_path)


@st.cache_data
def load_test_data():
    local_path = ensure_file_present("test_data.csv")
    return pd.read_csv(local_path)


st.title("❤️ Heart Disease Risk — Classification Demo")
st.write(
    "This app demonstrates 5 pre-trained machine learning classifiers for "
    "heart disease risk prediction. Upload a CSV of patient records (same "
    "format as `test_data.csv`) to get predictions and evaluation metrics."
)

st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
use_sample = st.sidebar.checkbox("Use bundled test_data.csv instead", value=uploaded_file is None)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif use_sample:
    data = load_test_data()
else:
    st.info("Upload a CSV or check 'Use bundled test_data.csv' to get started.")
    st.stop()

model = load_model(MODEL_FILES[model_name])

has_labels = TARGET in data.columns
X = data.drop(columns=[TARGET]) if has_labels else data.copy()

preds = model.predict(X)
proba = model.predict_proba(X)[:, 1]

results = data.copy()
results["predicted_" + TARGET] = preds
results["predicted_probability"] = proba.round(4)

st.subheader(f"Predictions — {model_name}")
st.dataframe(results.head(50), use_container_width=True)

col1, col2 = st.columns(2)

if has_labels:
    y_true = data[TARGET]
    metrics = {
        "Accuracy": accuracy_score(y_true, preds),
        "AUC": roc_auc_score(y_true, proba),
        "Precision": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "F1": f1_score(y_true, preds),
        "MCC": matthews_corrcoef(y_true, preds),
    }

    with col1:
        st.subheader("Evaluation Metrics")
        st.table(pd.DataFrame(metrics, index=[model_name]).round(4).T)

    with col2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 4))
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.subheader("Classification Report")
    st.code(classification_report(y_true, preds, target_names=["No Disease", "Disease"]))
else:
    st.warning(
        "No 'has_heart_disease' column found in the uploaded data, "
        "so evaluation metrics can't be computed — showing predictions only."
    )

st.subheader("Compare All Models")
if has_labels:
    all_rows = []
    for name, filename in MODEL_FILES.items():
        m = load_model(filename)
        p = m.predict(X)
        pr = m.predict_proba(X)[:, 1]
        all_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, p),
            "AUC": roc_auc_score(y_true, pr),
            "Precision": precision_score(y_true, p),
            "Recall": recall_score(y_true, p),
            "F1": f1_score(y_true, p),
            "MCC": matthews_corrcoef(y_true, p),
        })
    st.dataframe(pd.DataFrame(all_rows).round(4), use_container_width=True)
else:
    st.info("Upload data with true labels to compare all 5 models side by side.")

st.caption("Model comparison computed on the original held-out test split at training time:")
st.dataframe(load_comparison_table(), use_container_width=True)
