# Heart Disease Risk Prediction — ML Classification & Deployment

## Problem Statement
Cardiovascular disease is one of the leading causes of death worldwide, and early
risk identification from routine clinical and lifestyle measurements can help
prioritize preventive care. This project frames heart disease risk prediction as
a **binary classification problem**: given a patient's clinical readings (blood
pressure, cholesterol profile, blood sugar, etc.) and lifestyle indicators
(exercise, sleep, stress, smoking, diet), predict whether the patient has heart
disease (1) or not (0). Five classification models are
trained, evaluated, and compared, and the best-performing model is served
through an interactive Streamlit web app.

## Dataset Description
- **Source / file:** `heart_disease_risk_2026.csv`
- **Instances:** 9,000 patient records
- **Features:** 26 input features 
- **Target:** (0 = no disease, 1 = disease); class balance is ~70% / ~30%
- **Feature groups:**
  - *Clinical:* age, resting systolic/diastolic BP, total cholesterol, HDL, LDL, triglycerides, fasting blood sugar, HbA1c, BMI, resting/max heart rate, ST depression, chest pain type, exercise-induced angina, family history
  - *Lifestyle:* smoker status, alcohol units/week, exercise minutes/week, sleep hours, stress score, daily steps, diet quality score, wearable device ownership
- **Preprocessing:** boolean columns cast to 0/1, numeric features standardized , categorical features (`sex`, `chest_pain_type`, `smoker_status`) one-hot encoded, 80/20 stratified train/test split 

## GitHub Repository Link
(https://github.com/mrunalizes/2025ac05883-ML-Assignment)

## Models Used
All 5 models were trained on the same 80/20 stratified split of the dataset above.
Metrics were computed on the held-out 20% test set. Training code lives in
`model/train_models.py`; trained models are saved in `model/`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8978 | 0.9497 | 0.8632 | 0.7872 | 0.8234 | 0.7533 |
| Decision Tree | 0.8433 | 0.8952 | 0.7553 | 0.7138 | 0.7340 | 0.6236 |
| kNN | 0.8617 | 0.9150 | 0.8895 | 0.6202 | 0.7308 | 0.6606 |
| Naive Bayes | 0.8700 | 0.9329 | 0.7658 | 0.8220 | 0.7929 | 0.6993 |
| Random Forest (Ensemble) | 0.8872 | 0.9391 | 0.8800 | 0.7266 | 0.7960 | 0.7253 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer on this dataset — highest Accuracy, AUC, F1, and MCC. The relationship between the clinical/lifestyle features and heart disease risk appears largely linear/additive, which suits a linear decision boundary well, and the model generalizes cleanly without overfitting. |
| Decision Tree | Weakest of the five. A single tree with limited depth captures fewer of the additive relationships in the data and is more prone to variance, showing the lowest Accuracy, AUC, and MCC. |
| kNN | Highest Precision but the lowest Recall — it is conservative about flagging disease, missing more true positive cases than other models. Performance is sensitive to feature scaling and the choice of k. |
| Naive Bayes | Highest Recall among all models, making it useful when missing a true disease case is costlier than a false alarm, but its independence assumption between features caps its Precision and overall accuracy. |
| Random Forest (Ensemble) | Strong, well-rounded performer — close to Logistic Regression's AUC and MCC by averaging many trees to reduce the variance problem seen in the single Decision Tree, at the cost of some interpretability. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it gives the best balance of Accuracy, AUC, F1, and MCC, and is also the most interpretable and cheapest to serve in production. Random Forest is a strong runner-up if a non-linear model is preferred for future feature engineering. |

## Live Streamlit App Link
(https://2025ac05883-ml-assignment-2.streamlit.app/)

## Repository Structure
```
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
```

## How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates the 5 models in model/ + test_data.csv
streamlit run app.py
```

## App Features
- CSV upload of test data (or use the bundled `test_data.csv`)
- Model selection dropdown across the 5 trained models
- Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and full classification report
- Side-by-side comparison of all 5 models on the uploaded data

