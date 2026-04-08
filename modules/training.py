"""
EEG Full Pipeline: Preprocessing, Dataset Merging & Random Forest Classification
==================================================================================
Skripsi: Klasifikasi State Kognitif menggunakan Random Forest

Pipeline:
  BAGIAN 1: Preprocessing data noncognitive (raw EEG → PSD → noncognitive.csv)
  BAGIAN 2: Penyeragaman & penggabungan dataset → final_dataset.csv
  BAGIAN 3: Pembangunan model Random Forest → model_fix.pkl

Labels akhir:
  0 = Memory Recall
  1 = Arithmetic Calculation
  2 = Visual Pattern Recognition
  3 = Non Cognitive (go-nogo + resting)
"""

import pandas as pd
import numpy as np
from scipy.signal import resample, welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
ORIG_SR = 125           # Original sampling rate (Hz)
TARGET_SR = 512         # Target sampling rate (Hz)
WINDOW_SEC = 5          # Window size in seconds
WINDOW_SAMPLES = TARGET_SR * WINDOW_SEC  # 2560

# Selected channels (1-indexed) sesuai permintaan
SELECTED_CHANNELS = [1, 2, 5, 6, 10, 11, 23, 24, 27, 28, 8, 9, 16, 17, 21, 22]
FEATURE_COLS = [f"Channel_{ch}_PSD" for ch in SELECTED_CHANNELS]
ALL_COLS = FEATURE_COLS + ['Target_Label']

# Paths
GO_NOGO_DIR = r"E:\eeg-analysis\eeg-analysis\go nogo\go nogo"
RESTING_DIR = r"E:\eeg-analysis\eeg-analysis\resting\resting"
COGNITIVE_CSV = r"E:\eeg-analysis\eeg-analysis\cognitive_state_discrimination_dataset - cognitive_state_discrimination_dataset.csv"
NONCOGNITIVE_CSV = r"E:\eeg-analysis\eeg-analysis\noncognitive.csv"
FINAL_CSV = r"E:\eeg-analysis\eeg-analysis\final_dataset.csv"
MODEL_PATH = r"E:\eeg-analysis\modules\model_fix.pkl"

# Label names
LABEL_NAMES_DICT = {
    0: 'Memory Recall',
    1: 'Arithmetic Calculation',
    2: 'Visual Pattern',
    3: 'Non Cognitive',
}
LABEL_NAMES = ['Memory Recall', 'Arithmetic Calculation',
               'Visual Pattern', 'Non Cognitive']


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def load_raw_eeg(filepath):
    """
    Load raw OpenBCI EEG data.
    - Skip 4 baris header (komentar % dari OpenBCI)
    - Ambil 16 EXG channels saja
    """
    df = pd.read_csv(filepath, skiprows=4)
    df.columns = df.columns.str.strip()

    # Ambil hanya kolom EXG Channel 0 - 15
    eeg_cols = [col for col in df.columns if col.startswith('EXG Channel')]
    eeg_cols = sorted(eeg_cols, key=lambda x: int(x.split()[-1]))

    if len(eeg_cols) != 16:
        print(f"  WARNING: Expected 16 EXG channels, found {len(eeg_cols)} in {filepath}")

    data = df[eeg_cols].values.astype(np.float64)
    return data  # shape: (n_samples, 16)


def upsample_signal(data, orig_sr=ORIG_SR, target_sr=TARGET_SR):
    """Upsample signal dari orig_sr ke target_sr menggunakan scipy.signal.resample."""
    n_orig = data.shape[0]
    n_target = int(n_orig * target_sr / orig_sr)
    resampled = resample(data, n_target, axis=0)
    return resampled


def apply_windowing(data, sr=TARGET_SR, window_sec=WINDOW_SEC):
    """
    Segmentasi data ke dalam non-overlapping windows.
    Setiap window = sr * window_sec samples (2560 samples untuk 512Hz, 5 detik)
    """
    window_size = sr * window_sec
    n_windows = data.shape[0] // window_size
    windows = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        windows.append(data[start:end])
    return windows  # list of arrays, each shape (window_size, 16)


def compute_psd_features(window, sr=TARGET_SR):
    """
    Compute PSD features untuk satu window menggunakan Welch method.
    Returns: mean PSD per channel (single value per channel).
    """
    n_channels = window.shape[1]
    psd_values = []
    for ch in range(n_channels):
        freqs, psd = welch(window[:, ch], fs=sr, nperseg=min(512, window.shape[0]))
        psd_values.append(np.mean(psd))
    return psd_values


def process_subject(filepath):
    """
    Process satu subjek: load -> upsample -> window -> compute PSD.
    Returns: array shape (n_windows, 16)
    """
    data = load_raw_eeg(filepath)
    upsampled = upsample_signal(data)
    windows = apply_windowing(upsampled)

    psd_features = []
    for window in windows:
        psd = compute_psd_features(window)
        psd_features.append(psd)

    return np.array(psd_features)


# ============================================================
# BAGIAN 1: PREPROCESSING RAW EEG DATA -> noncognitive.csv
# ============================================================
print("=" * 60)
print("BAGIAN 1: PREPROCESSING RAW EEG DATA")
print("=" * 60)

all_psd_data = []

# Process go-nogo files (15 subjek)
go_nogo_files = sorted([f for f in os.listdir(GO_NOGO_DIR) if f.endswith('.txt')])
print(f"\nProcessing Go-NoGo files ({len(go_nogo_files)} subjects):")
for fname in go_nogo_files:
    filepath = os.path.join(GO_NOGO_DIR, fname)
    print(f"  Processing: {fname}...", end=" ", flush=True)
    psd_data = process_subject(filepath)
    all_psd_data.append(psd_data)
    print(f"-> {psd_data.shape[0]} windows extracted")

# Process resting files (10 subjek)
resting_files = sorted([f for f in os.listdir(RESTING_DIR) if f.endswith('.txt')])
print(f"\nProcessing Resting files ({len(resting_files)} subjects):")
for fname in resting_files:
    filepath = os.path.join(RESTING_DIR, fname)
    print(f"  Processing: {fname}...", end=" ", flush=True)
    psd_data = process_subject(filepath)
    all_psd_data.append(psd_data)
    print(f"-> {psd_data.shape[0]} windows extracted")

# Combine semua PSD data
combined_psd = np.vstack(all_psd_data)
print(f"\nTotal non-cognitive samples: {combined_psd.shape[0]}")

# Buat DataFrame dengan nama kolom yang sesuai
noncognitive_df = pd.DataFrame(combined_psd, columns=FEATURE_COLS)
noncognitive_df['Target_Label'] = 3  # Non Cognitive

# Save ke CSV
noncognitive_df.to_csv(NONCOGNITIVE_CSV, index=False)
print(f"[OK] Saved to: {NONCOGNITIVE_CSV}")
print(f"  Shape: {noncognitive_df.shape}")


# ============================================================
# BAGIAN 2: PENYERAGAMAN DATASET
# ============================================================
print("\n" + "=" * 60)
print("BAGIAN 2: PENYERAGAMAN DATASET")
print("=" * 60)


# --- Step 2.1: Load & format cognitive dataset ---
print("\n--- Step 2.1: Load & format cognitive dataset ---")
cognitive_df = pd.read_csv(COGNITIVE_CSV)
print(f"Original cognitive dataset shape: {cognitive_df.shape}")
print(f"Original columns ({len(cognitive_df.columns)}): {list(cognitive_df.columns[:5])} ... {list(cognitive_df.columns[-3:])}")

# Pastikan hanya kolom yang diperlukan yang diambil
cognitive_df = cognitive_df[ALL_COLS].copy()
print(f"After selecting required columns: {cognitive_df.shape}")
print(f"Columns: {list(cognitive_df.columns)}")


# --- Step 2.2: Remove label 3 from cognitive ---
print("\n--- Step 2.2: Remove label 3 from cognitive ---")
print(f"Label distribution BEFORE removal:")
for label_val in sorted(cognitive_df['Target_Label'].unique()):
    count = (cognitive_df['Target_Label'] == label_val).sum()
    label_name = LABEL_NAMES_DICT.get(label_val, f'Unknown ({label_val})')
    print(f"  Label {label_val} ({label_name}): {count} samples")

cognitive_df = cognitive_df[cognitive_df['Target_Label'] != 3].copy()
print(f"\nAfter removing label 3: {cognitive_df.shape}")

print(f"Label distribution AFTER removal:")
for label_val in sorted(cognitive_df['Target_Label'].unique()):
    count = (cognitive_df['Target_Label'] == label_val).sum()
    label_name = LABEL_NAMES_DICT.get(label_val, f'Unknown ({label_val})')
    print(f"  Label {label_val} ({label_name}): {count} samples")


# --- Step 2.3: Calculate & balance data per label ---
print("\n--- Step 2.3: Calculate & balance data per label ---")
total_cognitive = len(cognitive_df)
n_labels_cognitive = cognitive_df['Target_Label'].nunique()
data_per_label = total_cognitive // n_labels_cognitive
print(f"Total data setelah hapus label 3: {total_cognitive}")
print(f"Jumlah label tersisa: {n_labels_cognitive}")
print(f"Jumlah data per label: {total_cognitive} / {n_labels_cognitive} = {data_per_label}")

balanced_cognitive = []
for label_val in sorted(cognitive_df['Target_Label'].unique()):
    label_data = cognitive_df[cognitive_df['Target_Label'] == label_val]
    if len(label_data) >= data_per_label:
        sampled = label_data.sample(n=data_per_label, random_state=42)
    else:
        sampled = label_data
        print(f"  WARNING: Label {label_val} hanya punya {len(label_data)} samples")
    balanced_cognitive.append(sampled)
    print(f"  Label {label_val} ({LABEL_NAMES_DICT.get(label_val, '?')}): {len(sampled)} samples")

cognitive_balanced = pd.concat(balanced_cognitive, ignore_index=True)
print(f"Balanced cognitive dataset: {cognitive_balanced.shape}")


# --- Step 2.4: Prepare noncognitive (sample & relabel) ---
print("\n--- Step 2.4: Prepare noncognitive dataset ---")
noncognitive_df = pd.read_csv(NONCOGNITIVE_CSV)
print(f"Noncognitive dataset shape: {noncognitive_df.shape}")

noncognitive_df = noncognitive_df[ALL_COLS].copy()
noncognitive_sampled = noncognitive_df.sample(n=data_per_label, random_state=42).copy()
noncognitive_sampled['Target_Label'] = 3
print(f"Sampled noncognitive data: {noncognitive_sampled.shape}")
print(f"  Label: All set to 3 (Non Cognitive), Count: {len(noncognitive_sampled)}")


# --- Step 2.5: Verify column format ---
print("\n--- Step 2.5: Verify column format match ---")
cog_cols = list(cognitive_balanced.columns)
noncog_cols = list(noncognitive_sampled.columns)

if cog_cols == noncog_cols:
    print("[OK] Columns match perfectly!")
else:
    print("[WARNING] Column mismatch detected!")
    print(f"  Cognitive:    {cog_cols}")
    print(f"  Noncognitive: {noncog_cols}")


# --- Step 2.6: Merge & save final dataset ---
print("\n--- Step 2.6: Merge & save final dataset ---")
final_df = pd.concat([cognitive_balanced, noncognitive_sampled], ignore_index=True)
print(f"Final dataset shape: {final_df.shape}")

print(f"\nFinal label distribution:")
for label_val in sorted(final_df['Target_Label'].unique()):
    count = (final_df['Target_Label'] == label_val).sum()
    pct = count / len(final_df) * 100
    label_name = LABEL_NAMES_DICT.get(label_val, f'Unknown ({label_val})')
    print(f"  Label {label_val} ({label_name}): {count} samples ({pct:.1f}%)")
print(f"Total samples: {len(final_df)}")

final_df.to_csv(FINAL_CSV, index=False)
print(f"\n[OK] Final dataset saved to: {FINAL_CSV}")

# Verifikasi
verify_df = pd.read_csv(FINAL_CSV)
print(f"Verification - re-read: shape={verify_df.shape}, labels={sorted(verify_df['Target_Label'].unique())}")

print("\n" + "=" * 60)
print("BAGIAN 2 COMPLETE!")
print("=" * 60)


# ============================================================
# BAGIAN 3: PEMBANGUNAN MODEL RANDOM FOREST
# ============================================================
print("\n" + "=" * 60)
print("BAGIAN 3: PEMBANGUNAN MODEL RANDOM FOREST")
print("=" * 60)


# --- Step 3.1: Load final dataset ---
print("\n--- Step 3.1: Load final dataset ---")
df = pd.read_csv(FINAL_CSV)
print(f"Dataset shape: {df.shape}")

X = df[FEATURE_COLS]
y = df['Target_Label'].astype(int)

print(f"Features (X): {X.shape}")
print(f"Target  (y): {y.shape}")
print(f"\nLabel distribution:")
for label_val in sorted(y.unique()):
    count = (y == label_val).sum()
    pct = count / len(y) * 100
    print(f"  {LABEL_NAMES[label_val]} ({label_val}): {count} samples ({pct:.1f}%)")


# --- Step 3.2: Split data (70:30 stratified) ---
print("\n--- Step 3.2: Data split (70:30 stratified) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set:     {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nTraining set distribution:")
for label_val in sorted(y_train.unique()):
    count = (y_train == label_val).sum()
    pct = count / len(y_train) * 100
    print(f"  {LABEL_NAMES[label_val]} ({label_val}): {count} ({pct:.1f}%)")

print(f"\nTest set distribution:")
for label_val in sorted(y_test.unique()):
    count = (y_test == label_val).sum()
    pct = count / len(y_test) * 100
    print(f"  {LABEL_NAMES[label_val]} ({label_val}): {count} ({pct:.1f}%)")


# --- Step 3.3: Train Random Forest ---
print("\n--- Step 3.3: Train Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training model...")
rf_model.fit(X_train_scaled, y_train)

# Prediksi
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)

# Hitung akurasi
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
gap = train_acc - test_acc

print(f"\n[OK] Training Accuracy: {train_acc*100:.2f}%")
print(f"[OK] Test Accuracy:     {test_acc*100:.2f}%")
print(f"[OK] Train-Test Gap:    {gap*100:.2f}%")

print(f"\nModel Parameters:")
print(f"  n_estimators:      {rf_model.n_estimators}")
print(f"  max_depth:         {rf_model.max_depth}")
print(f"  min_samples_split: {rf_model.min_samples_split}")
print(f"  min_samples_leaf:  {rf_model.min_samples_leaf}")
print(f"  max_features:      {rf_model.max_features}")
print(f"  class_weight:      {rf_model.class_weight}")
print(f"  n_features_in_:    {rf_model.n_features_in_}")


# --- Step 3.4: Evaluation ---
print("\n--- Step 3.4: Detailed Evaluation ---")

precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

print("\n=== MODEL PERFORMANCE ===")
print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Gap:               {gap:.4f} ({gap*100:.2f}%)")
print(f"Precision:         {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:            {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:          {f1:.4f} ({f1*100:.2f}%)")

print(f"\n=== DATA SPLIT INFO ===")
print(f"X train: {X_train.shape} ({len(X_train)/len(X)*100:.1f}%)")
print(f"X test:  {X_test.shape} ({len(X_test)/len(X)*100:.1f}%)")
print(f"y train: {y_train.shape} ({len(y_train)} samples)")
print(f"y test:  {y_test.shape} ({len(y_test)} samples)")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_test_pred,
                            target_names=LABEL_NAMES, zero_division=0))

cm = confusion_matrix(y_test, y_test_pred)
print("=== CONFUSION MATRIX (Raw) ===")
print(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\n=== CONFUSION MATRIX (Normalized) ===")
np.set_printoptions(precision=3)
print(cm_normalized)

# Overfitting check
print("\n=== OVERFITTING CHECK ===")
if train_acc == 1.0:
    print("[WARNING] Training accuracy = 100%! Model mungkin overfitting.")
else:
    print(f"[OK] Training accuracy = {train_acc*100:.2f}% (bukan 100%)")

if gap > 0.15:
    print(f"[WARNING] Gap = {gap*100:.2f}% > 15%. Model mungkin overfitting.")
elif gap > 0.10:
    print(f"[CAUTION] Gap = {gap*100:.2f}% (10-15%). Perlu diperhatikan.")
else:
    print(f"[OK] Gap = {gap*100:.2f}% (<= 10%). Model cukup baik.")


# --- Step 3.5: Visualizations ---
print("\n--- Step 3.5: Generating Visualizations ---")

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# Plot 1: Raw confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            cbar_kws={'shrink': 0.8}, ax=axes[0])
axes[0].set_title('Confusion Matrix (Raw)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# Plot 2: Normalized confusion matrix
sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            cbar_kws={'shrink': 0.8}, ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

# Plot 3: Feature importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importances)))
axes[2].bar(range(len(importances)), importances[indices], color=colors)
axes[2].set_xticks(range(len(importances)))
axes[2].set_xticklabels([FEATURE_COLS[i] for i in indices], rotation=90, fontsize=8)
axes[2].set_title('Feature Importance', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Importance')
axes[2].set_xlabel('Features')

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(MODEL_PATH), 'model_evaluation.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[OK] Visualization saved to: {fig_path}")
plt.show()

print("\n=== FEATURE IMPORTANCE RANKING ===")
for i, idx in enumerate(indices):
    print(f"{i+1:2d}. {FEATURE_COLS[idx]:20s} - {importances[idx]:.4f}")


# --- Step 3.6: Save model ---
print("\n--- Step 3.6: Save model ---")

model_package = {
    'model': rf_model,
    'scaler': scaler,
    'feature_cols': FEATURE_COLS,
    'label_names': LABEL_NAMES,
    'selected_channels': SELECTED_CHANNELS,
    'metrics': {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'gap': gap,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
}

joblib.dump(model_package, MODEL_PATH)
print(f"[OK] Model saved to: {MODEL_PATH}")
print(f"\nFormat Model: {model_package}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DONE! Full pipeline complete.")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - Noncognitive data: {NONCOGNITIVE_CSV}")
print(f"  - Final dataset:     {FINAL_CSV}")
print(f"  - Model file:        {MODEL_PATH}")
print(f"  - Visualization:     {fig_path}")
print(f"\nModel Performance:")
print(f"  - Train Accuracy:    {train_acc*100:.2f}%")
print(f"  - Test Accuracy:     {test_acc*100:.2f}%")
print(f"  - Gap:               {gap*100:.2f}%")
print(f"  - Precision:         {precision*100:.2f}%")
print(f"  - Recall:            {recall*100:.2f}%")
print(f"  - F1-Score:          {f1*100:.2f}%")
