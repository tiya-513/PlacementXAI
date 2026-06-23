import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

import matplotlib.pyplot as plt


# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("dataset/student_placement_synthetic.csv")


# =====================================
# KEEP ONLY PLACED STUDENTS
# =====================================

salary_df = df[df["placement_status"] == 1].copy()


# =====================================
# FEATURES AND TARGET
# =====================================

X = salary_df.drop(
    ["placement_status", "salary_package_lpa"],
    axis=1
)

y = salary_df["salary_package_lpa"]


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
    random_state=42
)


# =====================================
# FEATURE SCALING
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =====================================
# BUILD NEURAL NETWORK
# =====================================

inputs = Input(shape=(X_train.shape[1],))

x = Dense(128, activation="relu")(inputs)
x = Dropout(0.2)(x)

x = Dense(64, activation="relu")(x)

x = Dense(32, activation="relu")(x)

salary_output = Dense(
    1,
    activation="linear",
    name="salary"
)(x)


# =====================================
# CREATE MODEL
# =====================================

model = Model(
    inputs=inputs,
    outputs=salary_output
)


# =====================================
# COMPILE MODEL
# =====================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# =====================================
# TRAIN MODEL
# =====================================
"""
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)
"""
model = load_model("salary_model.keras")

# =====================================
# EVALUATE MODEL
# =====================================

loss, mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n========================")
print("TEST RESULTS")
print("========================")
print(f"MSE Loss: {loss:.4f}")
print(f"MAE: {mae:.4f} LPA")


# =====================================
# SAMPLE PREDICTIONS
# =====================================

predictions = model.predict(X_test)

print("\n========================")
print("SAMPLE PREDICTIONS")
print("========================")

for i in range(10):

    print(
        f"Actual Salary: {y_test.iloc[i]:.2f} LPA | "
        f"Predicted Salary: {predictions[i][0]:.2f} LPA"
    )

r2 = r2_score(y_test, predictions)

print(f"R² Score: {r2:.4f}")

import joblib

joblib.dump(
    scaler,
    "salary_scaler.pkl"
)

# =====================================
# SAVE MODEL
# =====================================

model.save("salary_model.keras")

print("\nModel saved as salary_model.keras")
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n========================")
print("EVALUATION METRICS")
print("========================")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f} LPA")
print(f"R²   : {r2:.4f}")