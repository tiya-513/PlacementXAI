import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("student_placement_synthetic.csv")


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
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print(class_weights)

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

x = Dense(128, activation="tanh")(inputs)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(64, activation="tanh")(x)
x = BatchNormalization()(x)

x = Dense(32, activation="tanh")(x)
x = BatchNormalization()(x)

placement_output = Dense(
    1,
    activation="sigmoid",
    name="placement"
)(x)


# =====================================
# CREATE MODEL
# =====================================

model = Model(
    inputs=inputs,
    outputs=placement_output
)


# =====================================
# COMPILE MODEL
# =====================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =====================================
# TRAIN MODEL
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    callbacks=[early_stop],
    batch_size=32,
    class_weight=class_weights,
    verbose=1
)
#model = load_model("placement_model.keras")

# =====================================
# EVALUATE MODEL
# =====================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n========================")
print("TEST RESULTS")
print("========================")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")


# =====================================
# SAMPLE PREDICTIONS
# =====================================

predictions = model.predict(X_test)

predictions_binary = (
    predictions > 0.5
).astype(int)

print("\n========================")
print("SAMPLE PREDICTIONS")
print("========================")

for i in range(10):

    print(
        f"Actual: {y_test.iloc[i]} | "
        f"Predicted: {predictions_binary[i][0]} | "
        f"Probability: {predictions[i][0]:.4f}"
    )


# =====================================
# SAVE MODEL
# =====================================

model.save("placement_model.keras")

print("\nModel saved as placement_model.keras")

print(confusion_matrix(
    y_test,
    predictions_binary
))

print(classification_report(
    y_test,
    predictions_binary
))

probs = model.predict(X_test)

auc = roc_auc_score(y_test, probs)

print("ROC-AUC:", auc)



# Training and validation loss
plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss Curve")

plt.legend()
plt.show()