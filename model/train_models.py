"""
model/train_models.py
----------------------
Trains all 5 required classification models on the Heart Disease Risk (2026)
dataset a
 model/:
    logistic_regression.joblib
    decision_tree.joblib
    knn.joblib
    naive_bayes.joblib
    random_forest_ensemble.joblib
    metrics_comparison.csv

"""

import os
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))    
PROJECT_ROOT = os.path.dirname(THIS_DIR)               
DATA_PATH = os.path.join(PROJECT_ROOT, "heart_disease_risk_2026.csv")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "test_data.csv")

RANDOM_STATE = 42
TARGET = "has_heart_disease"

NUMERIC_FEATURES = [
    "age", "resting_bp_systolic", "resting_bp_diastolic", "cholesterol_total",
    "hdl", "ldl", "triglycerides", "fasting_blood_sugar", "hba1c", "bmi",
    "resting_heart_rate", "max_heart_rate_achieved", "st_depression",
    "alcohol_units_per_week", "exercise_minutes_per_week", "sleep_hours",
    "stress_score", "daily_steps", "diet_quality_score",
]
CATEGORICAL_FEATURES = ["sex", "chest_pain_type", "smoker_status"]
BOOLEAN_FEATURES = ["exercise_induced_angina", "family_history", "wearable_owner"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["patient_id"])
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES + BOOLEAN_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE
        ),
    }


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["Model"] = name
        results.append(metrics)

        out_path = os.path.join(THIS_DIR, MODEL_FILES[name])
        joblib.dump(pipe, out_path)
        print(f"Saved {name} -> {out_path}")

    results_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ].round(4)
    results_df.to_csv(os.path.join(THIS_DIR, "metrics_comparison.csv"), index=False)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    test_sample = X_test.copy()
    test_sample[TARGET] = y_test
    test_sample.sample(n=min(300, len(test_sample)), random_state=RANDOM_STATE).to_csv(
        TEST_DATA_PATH, index=False
    )
    print(f"\nSaved test_data.csv -> {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
