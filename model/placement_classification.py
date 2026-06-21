import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from xgboost import XGBClassifier


# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("dataset/student_placement_synthetic.csv")


# =====================================
# FEATURES AND TARGET
# =====================================

X = df.drop(
    ["placement_status", "salary_package_lpa"],
    axis=1
)

y = df["placement_status"]


# =====================================
# ENCODE CATEGORICAL FEATURES
# =====================================

X = pd.get_dummies(
    X,
    columns=["branch", "college_tier"],
    drop_first=True
)

for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)


# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# HANDLE CLASS IMBALANCE
# =====================================

neg, pos = np.bincount(y_train)

scale_pos_weight = neg / pos

print(f"scale_pos_weight = {scale_pos_weight:.4f}")


# =====================================
# XGBOOST MODEL
# =====================================

model = XGBClassifier(
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


# =====================================
# TRAIN MODEL
# =====================================

model.fit(X_train, y_train)


# =====================================
# PREDICTIONS
# =====================================

threshold = 0.43

probs = model.predict_proba(X_test)[:, 1]

preds = (probs > threshold).astype(int)


# =====================================
# METRICS
# =====================================

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, probs)

print("\n========================")
print("XGBOOST RESULTS")
print("========================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")


# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\n========================")
print("CLASSIFICATION REPORT")
print("========================")

print(classification_report(y_test, preds))


# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_test, preds)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Placed", "Placed"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.show()


# =====================================
# ROC CURVE
# =====================================

fpr, tpr, _ = roc_curve(y_test, probs)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random Guess"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.show()


# =====================================
# TOP 10 FEATURE IMPORTANCES
# =====================================

importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n========================")
print("TOP 10 FEATURES")
print("========================")

print(importances.head(10))

# Save model
model.save_model("placement_xgboost.json")

print("Model saved successfully!")