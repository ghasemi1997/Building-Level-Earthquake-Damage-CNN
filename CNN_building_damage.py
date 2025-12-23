import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# --------------------------------------------------
# 1. Load building-level dataset
# --------------------------------------------------
data = pd.read_csv("building_level_features.csv")

feature_columns = [
    "Intensity_Pre",
    "Intensity_Post",
    "Coherence_Pre",
    "Coherence_Post",
    "Geology",
    "PGA",
    "Vs30"
]

X = data[feature_columns]
y = data["Damage_Label"]  # 0 = undamaged, 1 = damaged

# --------------------------------------------------
# 2. Plot histograms (Feature distribution analysis)
# --------------------------------------------------
plt.figure(figsize=(14, 10))

for i, feature in enumerate(feature_columns):
    plt.subplot(3, 3, i + 1)
    plt.hist(X[y == 0][feature], bins=30, alpha=0.6, label="Undamaged")
    plt.hist(X[y == 1][feature], bins=30, alpha=0.6, label="Damaged")
    plt.title(feature)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig("feature_histograms.png", dpi=300)
plt.show()

# --------------------------------------------------
# 3. Feature normalization
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 4. Reshape for Conv1D
# Each building -> (7 features, 1 channel)
# --------------------------------------------------
X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
y_cat = to_categorical(y, num_classes=2)

# --------------------------------------------------
# 5. Train / validation split
# --------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_cnn, y_cat,
    test_size=0.5,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 6. CNN model (feature-based)
# --------------------------------------------------
model = Sequential([
    Conv1D(64, kernel_size=1, activation="relu", input_shape=(7, 1)),
    MaxPooling1D(pool_size=1),
    Conv1D(32, kernel_size=1, activation="relu"),
    Conv1D(16, kernel_size=1, activation="relu"),
    Flatten(),
    Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# 7. Training
# --------------------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=150,
    validation_data=(X_val, y_val),
    verbose=1
)

# --------------------------------------------------
# 8. Evaluation
# --------------------------------------------------
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {accuracy:.4f}")
# --------------------------------------------------
# 9. Plot training and validation loss & accuracy
# --------------------------------------------------

# Accuracy plot
plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300)
plt.show()

# Loss plot
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.show()
