"""
Noncognitive Preprocessing Pipeline
====================================
Pipeline untuk data mentah EEG (Go-NoGo + Resting State):
  1) Load file .txt OpenBCI, skip 4 baris metadata
  2) Upsampling 125 Hz → 512 Hz (scipy.signal.resample)
  3) Windowing non-overlap 2 detik (1024 sampel @ 512 Hz)
  4) Ekstraksi fitur PSD (Welch) per channel → mean power
  5) Scaling Min-Max (0-1) — scaler di-fit pada gabungan noncognitive
  6) Simpan output:
       - E:/eeg-analysis/eeg-analysis/gonogo.csv   (15 subjek)
       - E:/eeg-analysis/eeg-analysis/resting.csv  (10 subjek)
       - E:/eeg-analysis/eeg-analysis/noncognitive.csv (gabungan)

Dataset Go-NoGo : E:/eeg-analysis/eeg-analysis/go nogo/go nogo/
Dataset Resting : E:/eeg-analysis/eeg-analysis/resting/resting/
"""

import os
from glob import glob

import numpy as np
import pandas as pd
from scipy.signal import resample, welch
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# CONFIGURATION
# ============================================================
ORIG_SR      = 125          # Sample rate asli OpenBCI (Hz)
TARGET_SR    = 512          # Target sample rate setelah upsampling
WINDOW_SEC   = 2            # Durasi window (detik)
WINDOW_SAMPLES = TARGET_SR * WINDOW_SEC  # = 1024 sampel per window

# 16 channel yang digunakan (sesuai dataset kognitif)
SELECTED_CHANNELS = [1, 2, 5, 6, 10, 11, 23, 24, 27, 28, 8, 9, 16, 17, 21, 22]
FEATURE_COLS = [f"Channel_{ch}_PSD" for ch in SELECTED_CHANNELS]

# Path data mentah — data ada di subfolder dengan nama sama
GO_NOGO_DIR    = r"E:\eeg-analysis\eeg-analysis\go nogo\go nogo"
RESTING_DIR    = r"E:\eeg-analysis\eeg-analysis\resting\resting"

# Path output
OUT_GONOGO        = r"E:\eeg-analysis\eeg-analysis\gonogo.csv"
OUT_RESTING       = r"E:\eeg-analysis\eeg-analysis\resting.csv"
OUT_NONCOGNITIVE  = r"E:\eeg-analysis\eeg-analysis\noncognitive.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def discover_txt_files(root_dir: str) -> list[str]:
    """Cari semua file .txt secara rekursif dalam direktori."""
    pattern = os.path.join(root_dir, "**", "*.txt")
    files = sorted(glob(pattern, recursive=True))
    if not files:
        # Fallback: cari langsung di root_dir (tidak rekursif)
        files = sorted(glob(os.path.join(root_dir, "*.txt")))
    return files


def load_raw_eeg(filepath: str) -> np.ndarray:
    """
    Load file .txt OpenBCI:
    - Skip 4 baris metadata
    - Ambil kolom 'EXG Channel 0' .. 'EXG Channel 15' (16 kanal)
    Return: numpy array shape (n_samples, 16), dtype float64
    """
    df = pd.read_csv(filepath, skiprows=4, low_memory=False)
    df.columns = df.columns.str.strip()

    # Deteksi kolom EXG Channel
    eeg_cols = [col for col in df.columns if col.strip().startswith("EXG Channel")]
    eeg_cols = sorted(eeg_cols, key=lambda x: int(x.strip().split()[-1]))

    if len(eeg_cols) < 16:
        raise ValueError(
            f"Hanya ditemukan {len(eeg_cols)} EXG Channel (butuh 16) di: {filepath}"
        )

    # Gunakan 16 channel pertama kalau ada lebih
    eeg_cols = eeg_cols[:16]
    data = df[eeg_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)

    # Hapus baris yang semua nilainya NaN
    mask = ~np.isnan(data).all(axis=1)
    data = data[mask]

    # Interpolasi NaN individual (sederhana: fill forward)
    for ch in range(data.shape[1]):
        col = pd.Series(data[:, ch])
        col = col.ffill().bfill()
        data[:, ch] = col.values

    return data


def upsample_signal(data: np.ndarray,
                    orig_sr: int = ORIG_SR,
                    target_sr: int = TARGET_SR) -> np.ndarray:
    """Upsample dari orig_sr → target_sr menggunakan scipy.signal.resample."""
    n_orig   = data.shape[0]
    n_target = int(round(n_orig * target_sr / orig_sr))
    return resample(data, n_target, axis=0)


def apply_windowing(data: np.ndarray,
                    sr: int = TARGET_SR,
                    window_sec: int = WINDOW_SEC) -> list[np.ndarray]:
    """
    Potong data menjadi windows non-overlapping.
    Return: list of windows, masing-masing shape (window_samples, n_channels)
    """
    window_size = sr * window_sec
    n_windows   = data.shape[0] // window_size
    return [data[i * window_size:(i + 1) * window_size] for i in range(n_windows)]


def compute_psd_features(window: np.ndarray,
                         sr: int = TARGET_SR) -> list[float]:
    """
    Hitung mean PSD (Welch) untuk setiap channel dalam satu window.
    Return: list of float, panjang = jumlah channel (16)
    """
    psd_values = []
    for ch in range(window.shape[1]):
        _, psd = welch(window[:, ch], fs=sr, nperseg=min(256, window.shape[0]))
        psd_values.append(float(np.mean(psd)))
    return psd_values


def participant_id_from_path(filepath: str) -> str:
    """Ambil nama file tanpa ekstensi sebagai participant ID."""
    return os.path.splitext(os.path.basename(filepath))[0]


def process_group(txt_files: list[str], task_name: str) -> pd.DataFrame:
    """
    Proses semua file .txt dalam satu grup (Go-NoGo atau Resting):
      load → upsample → window → PSD → DataFrame (belum di-scale)
    """
    rows = []
    for filepath in txt_files:
        pid = participant_id_from_path(filepath)
        fname = os.path.basename(filepath)
        print(f"    Processing [{task_name}] {fname} ...", end=" ", flush=True)

        try:
            raw       = load_raw_eeg(filepath)
            upsampled = upsample_signal(raw)
            windows   = apply_windowing(upsampled)

            for w in windows:
                psd = compute_psd_features(w)
                row = {
                    "Participant_ID": pid,
                    "Task":           task_name,
                    "Target_Label":   3,          # Non-Cognitive = label 3
                }
                row.update({FEATURE_COLS[i]: psd[i] for i in range(len(FEATURE_COLS))})
                rows.append(row)

            print(f"OK ({len(windows)} windows)")

        except Exception as exc:
            print(f"ERROR — {exc}")

    ordered_cols = ["Participant_ID", "Task"] + FEATURE_COLS + ["Target_Label"]
    if not rows:
        return pd.DataFrame(columns=ordered_cols)

    return pd.DataFrame(rows)[ordered_cols]


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("NONCOGNITIVE EEG PREPROCESSING")
    print("=" * 60)
    print(f"  Upsampling : {ORIG_SR} Hz  →  {TARGET_SR} Hz")
    print(f"  Window     : {WINDOW_SEC} detik  ({WINDOW_SAMPLES} sampel)")
    print(f"  Channels   : {len(SELECTED_CHANNELS)} channel")
    print(f"  PSD method : Welch (nperseg=256)")
    print(f"  Scaling    : MinMax [0, 1]")
    print()

    # --- Discover files ---
    go_nogo_files  = discover_txt_files(GO_NOGO_DIR)
    resting_files  = discover_txt_files(RESTING_DIR)

    print(f"Go-NoGo dir  : {GO_NOGO_DIR}")
    print(f"  Files found: {len(go_nogo_files)}")
    print(f"Resting dir  : {RESTING_DIR}")
    print(f"  Files found: {len(resting_files)}")
    print()

    if not go_nogo_files and not resting_files:
        print("[ERROR] Tidak ada file .txt yang ditemukan. Periksa path direktori.")
        return

    # --- Process each group ---
    print("[1/4] Processing Go-NoGo data ...")
    go_nogo_df = process_group(go_nogo_files, "Go-NoGo")

    print(f"\n[2/4] Processing Resting State data ...")
    resting_df = process_group(resting_files, "Resting State")

    print(f"\n  Go-NoGo  : {go_nogo_df.shape[0]} windows dari {len(go_nogo_files)} file")
    print(f"  Resting  : {resting_df.shape[0]} windows dari {len(resting_files)} file")

    # --- Concat ---
    print("\n[3/4] Menggabungkan & menerapkan MinMax Scaling (0–1) ...")
    combined_df = pd.concat([go_nogo_df, resting_df], ignore_index=True)

    if len(combined_df) == 0:
        print("[ERROR] Tidak ada data yang berhasil diproses.")
        return

    # Fit scaler pada seluruh noncognitive data, transform masing-masing subset
    scaler = MinMaxScaler(feature_range=(0, 1))
    combined_df[FEATURE_COLS] = scaler.fit_transform(combined_df[FEATURE_COLS])
    combined_df[FEATURE_COLS] = np.clip(combined_df[FEATURE_COLS], 0.0, 1.0)

    # Sinkronkan nilai scaled ke subset masing-masing
    if len(go_nogo_df) > 0:
        go_nogo_idx = combined_df.index[: len(go_nogo_df)]
        go_nogo_df[FEATURE_COLS] = combined_df.loc[go_nogo_idx, FEATURE_COLS].values
        go_nogo_df[FEATURE_COLS] = np.clip(go_nogo_df[FEATURE_COLS], 0.0, 1.0)

    if len(resting_df) > 0:
        resting_idx = combined_df.index[len(go_nogo_df):]
        resting_df[FEATURE_COLS] = combined_df.loc[resting_idx, FEATURE_COLS].values
        resting_df[FEATURE_COLS] = np.clip(resting_df[FEATURE_COLS], 0.0, 1.0)

    # --- Save ---
    print(f"\n[4/4] Menyimpan CSV ...")
    go_nogo_df.to_csv(OUT_GONOGO,       index=False)
    resting_df.to_csv(OUT_RESTING,      index=False)
    combined_df.to_csv(OUT_NONCOGNITIVE, index=False)

    print(f"\n[OK] File berhasil disimpan:")
    print(f"  {OUT_GONOGO:<60}  rows={go_nogo_df.shape[0]}")
    print(f"  {OUT_RESTING:<60}  rows={resting_df.shape[0]}")
    print(f"  {OUT_NONCOGNITIVE:<60}  rows={combined_df.shape[0]}")

    vmin = float(combined_df[FEATURE_COLS].min().min())
    vmax = float(combined_df[FEATURE_COLS].max().max())
    print(f"\n  Nilai fitur setelah scaling: min={vmin:.6f}  max={vmax:.6f}")
    print("=" * 60)
    print("DONE: Preprocessing selesai")
    print("=" * 60)


if __name__ == "__main__":
    main()
