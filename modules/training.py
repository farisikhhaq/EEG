"""
Model Training Pipeline (Training Only)
======================================
File ini khusus untuk training model Random Forest dari dataset final yang sudah siap.

Input utama:
- E:/eeg-analysis/eeg-analysis/final_dataset.csv

Output:
- E:/eeg-analysis/modules/model_fix.pkl
- E:/eeg-analysis/modules/model_evaluation.png
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================
SELECTED_CHANNELS = [1, 2, 5, 6, 10, 11, 23, 24, 27, 28, 8, 9, 16, 17, 21, 22]
REAL_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
FEATURE_COLS = [f"Channel_{ch}_PSD" for ch in REAL_CHANNELS]
FINAL_CSV = r"E:\eeg-analysis\eeg-analysis\final_dataset.csv"
MODEL_PATH = r"E:\eeg-analysis\modules\model_fix.pkl"

LABEL_NAMES = ["Memory Recall", "Arithmetic Calculation", "Visual Pattern", "Non Cognitive"]


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("MODEL TRAINING ONLY")
print("=" * 60)

if not os.path.exists(FINAL_CSV):
    raise FileNotFoundError(f"Final dataset tidak ditemukan: {FINAL_CSV}")

df = pd.read_csv(FINAL_CSV)
required_cols = FEATURE_COLS + ["Target_Label"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Kolom wajib tidak ditemukan di final dataset: {missing_cols}")

X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df["Target_Label"], errors="coerce").astype("Int64")

valid_mask = (~X.isna().any(axis=1)) & (y.notna())
X = X.loc[valid_mask].astype(float)
y = y.loc[valid_mask].astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Valid rows for training: {len(X)}")
print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

print("\nLabel distribution:")
for label_val in sorted(y.unique()):
    count = int((y == label_val).sum())
    pct = count / len(y) * 100
    label_name = LABEL_NAMES[label_val] if 0 <= label_val < len(LABEL_NAMES) else str(label_val)
    print(f"  {label_name} ({label_val}): {count} samples ({pct:.1f}%)")


# ============================================================
# TRAIN-TEST SPLIT + SCALING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled[0:5])  # Contoh data yang sudah diskalakan

print("\nData split:")
print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")


# ============================================================
# MODEL TRAINING
# ============================================================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("\nTraining Random Forest...")
rf_model.fit(X_train_scaled, y_train)

y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
gap = train_acc - test_acc
precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

print("\n=== MODEL PERFORMANCE ===")
print(f"Training Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
print(f"Test Accuracy:     {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"Gap:               {gap:.4f} ({gap * 100:.2f}%)")
print(f"Precision:         {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:            {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-Score:          {f1:.4f} ({f1 * 100:.2f}%)")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_test_pred, target_names=LABEL_NAMES, zero_division=0))

cm = confusion_matrix(y_test, y_test_pred)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]


# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES,
    cbar_kws={"shrink": 0.8},
    ax=axes[0],
)
axes[0].set_title("Confusion Matrix (Raw)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted Label")
axes[0].tick_params(axis="x", rotation=45)
axes[0].tick_params(axis="y", rotation=0)

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES,
    cbar_kws={"shrink": 0.8},
    ax=axes[1],
)
axes[1].set_title("Confusion Matrix (Normalized)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("True Label")
axes[1].set_xlabel("Predicted Label")
axes[1].tick_params(axis="x", rotation=45)
axes[1].tick_params(axis="y", rotation=0)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importances)))
axes[2].bar(range(len(importances)), importances[indices], color=colors)
axes[2].set_xticks(range(len(importances)))
axes[2].set_xticklabels([FEATURE_COLS[i] for i in indices], rotation=90, fontsize=8)
axes[2].set_title("Feature Importance", fontsize=12, fontweight="bold")
axes[2].set_ylabel("Importance")
axes[2].set_xlabel("Features")

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(MODEL_PATH), "model_evaluation.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n[OK] Visualization saved to: {fig_path}")
plt.show()


# ============================================================
# SAVE MODEL
# ============================================================
model_package = {
    "model": rf_model,
    "scaler": scaler,
    "feature_cols": FEATURE_COLS,
    "label_names": LABEL_NAMES,
    "selected_channels": SELECTED_CHANNELS,
    "metrics": {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "gap": gap,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    },
}
joblib.dump(model_package, MODEL_PATH)

print(f"[OK] Model saved to: {MODEL_PATH}")
print("=" * 60)
print("DONE: Training complete")
print("=" * 60)
