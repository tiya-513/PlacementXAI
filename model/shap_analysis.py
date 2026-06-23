import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model


# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("dataset/student_placement_synthetic.csv")


# ==========================================================
# PLACEMENT MODEL (XGBOOST)
# ==========================================================

print("\n" + "=" * 60)
print("PLACEMENT MODEL SHAP ANALYSIS")
print("=" * 60)

X_place = df.drop(
    ["placement_status", "salary_package_lpa"],
    axis=1
)

y_place = df["placement_status"]

X_place = pd.get_dummies(
    X_place,
    columns=["branch", "college_tier"],
    drop_first=True
)

for col in X_place.columns:
    if X_place[col].dtype == bool:
        X_place[col] = X_place[col].astype(int)


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train_place, X_test_place, y_train_place, y_test_place = train_test_split(
    X_place,
    y_place,
    test_size=0.2,
    random_state=42,
    stratify=y_place
)


# ==========================================================
# HANDLE CLASS IMBALANCE
# ==========================================================

neg, pos = np.bincount(y_train_place)

scale_pos_weight = neg / pos


# ==========================================================
# TRAIN XGBOOST MODEL
# ==========================================================

placement_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost model...")

placement_model.fit(
    X_train_place,
    y_train_place
)

print("Training complete.")


# ==========================================================
# SAMPLE DATA FOR SHAP
# ==========================================================

sample_size = 200

X_place_sample = X_place.sample(
    sample_size,
    random_state=42
)

print("Creating SHAP Permutation Explainer...")

placement_explainer = shap.PermutationExplainer(
    placement_model.predict_proba,
    X_place_sample
)

print("Calculating SHAP values...")
print("This may take several minutes...")

placement_shap_values = placement_explainer(
    X_place_sample
)


# ==========================================================
# PLACEMENT SUMMARY PLOT
# ==========================================================

print("Generating Placement Summary Plot...")

shap.plots.beeswarm(
    placement_shap_values[:, :, 1],
    max_display=20
)

plt.show()


# ==========================================================
# PLACEMENT FEATURE IMPORTANCE
# ==========================================================

print("Generating Placement Feature Importance Plot...")

shap.plots.bar(
    placement_shap_values[:, :, 1],
    max_display=20
)

plt.show()


# ==========================================================
# SINGLE STUDENT EXPLANATION
# ==========================================================

student_idx = 0

print("Generating Placement Waterfall Plot...")

shap.plots.waterfall(
    placement_shap_values[student_idx, :, 1],
    max_display=15
)

plt.show()


# ==========================================================
# SALARY MODEL (NEURAL NETWORK)
# ==========================================================

print("\n" + "=" * 60)
print("SALARY MODEL SHAP ANALYSIS")
print("=" * 60)

salary_df = df[
    df["placement_status"] == 1
].copy()

X_salary = salary_df.drop(
    ["placement_status", "salary_package_lpa"],
    axis=1
)

X_salary = pd.get_dummies(
    X_salary,
    columns=["branch", "college_tier"],
    drop_first=True
)

for col in X_salary.columns:
    if X_salary[col].dtype == bool:
        X_salary[col] = X_salary[col].astype(int)


# ==========================================================
# LOAD SAVED SCALER
# ==========================================================

salary_scaler = joblib.load(
    "salary_scaler.pkl"
)

X_salary_scaled = salary_scaler.transform(
    X_salary
)


# ==========================================================
# LOAD SALARY MODEL
# ==========================================================

salary_model = load_model(
    "salary_model.keras"
)

print("Salary model loaded.")


# ==========================================================
# BACKGROUND DATA
# ==========================================================

background = X_salary_scaled[
    np.random.choice(
        X_salary_scaled.shape[0],
        50,
        replace=False
    )
]


# ==========================================================
# KERNEL EXPLAINER
# ==========================================================

print("Creating KernelExplainer...")

salary_explainer = shap.KernelExplainer(
    salary_model.predict,
    background
)


# ==========================================================
# SAMPLE DATA TO EXPLAIN
# ==========================================================

sample_data = X_salary_scaled[:25]

print("Calculating SHAP values for salary model...")
print("This may take a few minutes...")

salary_shap_values = salary_explainer.shap_values(
    sample_data
)


# ==========================================================
# SALARY SUMMARY PLOT
# ==========================================================

print("Generating Salary Summary Plot...")

shap.summary_plot(
    salary_shap_values,
    sample_data,
    feature_names=X_salary.columns
)


# ==========================================================
# SALARY BAR PLOT
# ==========================================================

print("Generating Salary Feature Importance Plot...")

shap.summary_plot(
    salary_shap_values,
    sample_data,
    feature_names=X_salary.columns,
    plot_type="bar"
)


# ==========================================================
# SINGLE STUDENT SALARY EXPLANATION
# ==========================================================

student_idx = 0

print("Generating Salary Force Plot...")

shap.force_plot(
    salary_explainer.expected_value,
    salary_shap_values[student_idx],
    sample_data[student_idx],
    feature_names=X_salary.columns,
    matplotlib=True
)

plt.show()


# ==========================================================
# COMPLETE
# ==========================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETE")
print("=" * 60)