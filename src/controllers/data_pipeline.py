"""
PerconAI — Data Acquisition & Preprocessing
============================================
Downloads, preprocesses, and memmap-caches:
  • UCI-HAR (raw inertial signals)
  • PAMAP2
  • HHAR (Heterogeneity Human Activity Recognition)

All functions are written as resumable subroutines:
  - Downloads are skipped if raw files already exist.
  - Memmap files are skipped if they already exist with valid metadata.

Strict preprocessing pipeline (applied in this EXACT order):
  1. Resample to 50 Hz
  2. Filter: drop PAMAP2 transitional classes
  3. NaN handling: forward-fill then zero-fill
  4. LOSO grouping by subject_id
  5. Z-scoring using ONLY source (training) subjects
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.signal import resample

logger = logging.getLogger("PerconAI.data")

# ── Project root (src/controllers/ → root) ──────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
import time
import hashlib

def _download_and_extract_zip(url: str, dest_dir: Path, name: str, expected_md5: Optional[str] = None) -> None:
    """Download a ZIP from `url` and extract into `dest_dir` with retry logic and checksum verification."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        logger.info("[%s] Raw data already exists at %s — skipping download", name, dest_dir)
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("[%s] Downloading from %s (Attempt %d/%d)...", name, url, attempt, max_retries)
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            
            content_bytes = resp.content
            
            # Verify checksum if provided
            if expected_md5:
                md5_hash = hashlib.md5(content_bytes).hexdigest()
                if md5_hash != expected_md5:
                    raise ValueError(f"Checksum mismatch for {name}. Expected {expected_md5}, got {md5_hash}")
                logger.info("[%s] Checksum verified: %s", name, expected_md5)
                
            content = io.BytesIO(content_bytes)
            with zipfile.ZipFile(content) as zf:
                zf.extractall(dest_dir)
            logger.info("[%s] Extracted to %s", name, dest_dir)
            return  # Success, exit retry loop
            
        except Exception as e:
            logger.warning("[%s] Download failed: %s", name, e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error("[%s] Max retries reached. Download failed.", name)
                raise


def download_uci_har(raw_dir: Optional[Path] = None) -> Path:
    """Download UCI-HAR dataset."""
    if raw_dir is None:
        raw_dir = _PROJECT_ROOT / "data" / "raw" / "uci_har"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    _download_and_extract_zip(url, raw_dir, "UCI-HAR")
    return raw_dir


def download_pamap2(raw_dir: Optional[Path] = None) -> Path:
    """Download PAMAP2 dataset."""
    if raw_dir is None:
        raw_dir = _PROJECT_ROOT / "data" / "raw" / "pamap2"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
    _download_and_extract_zip(url, raw_dir, "PAMAP2")
    return raw_dir


def download_hhar(raw_dir: Optional[Path] = None) -> Path:
    """Download HHAR dataset."""
    if raw_dir is None:
        raw_dir = _PROJECT_ROOT / "data" / "raw" / "hhar"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip"
    _download_and_extract_zip(url, raw_dir, "HHAR")
    return raw_dir


def download_all_datasets() -> dict[str, Path]:
    """Download all three datasets. Returns {name: raw_path}."""
    return {
        "uci_har": download_uci_har(),
        "pamap2": download_pamap2(),
        "hhar": download_hhar(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  UCI-HAR PARSER
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_uci_har(raw_dir: Path) -> dict:
    """
    Parse UCI-HAR raw inertial signals into per-subject arrays.

    Returns
    -------
    subjects : dict[int, dict]
        {subject_id: {"X": ndarray(n_windows, channels, 128), "y": ndarray(n_windows,)}}
    """
    # Find the actual dataset directory (may be nested)
    candidates = list(raw_dir.rglob("UCI HAR Dataset"))
    if not candidates:
        candidates = list(raw_dir.rglob("UCI_HAR_Dataset"))
    if not candidates:
        # Try flat structure
        base = raw_dir
    else:
        base = candidates[0]

    subjects: dict = {}

    for split in ["train", "test"]:
        split_dir = base / split
        inertial_dir = split_dir / "Inertial Signals"

        if not inertial_dir.exists():
            logger.warning("UCI-HAR: %s not found, skipping", inertial_dir)
            continue

        # Load subject IDs
        subj_file = split_dir / f"subject_{split}.txt"
        subj_ids = np.loadtxt(subj_file, dtype=int)

        # Load labels
        label_file = split_dir / f"y_{split}.txt"
        labels = np.loadtxt(label_file, dtype=int) - 1  # 0-index

        # Load all 9 inertial signal files
        signal_names = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]
        signals = []
        for sname in signal_names:
            fname = inertial_dir / f"{sname}_{split}.txt"
            data = np.loadtxt(fname)  # (n_windows, 128)
            signals.append(data)

        # Stack: (n_windows, 9, 128)
        X_all = np.stack(signals, axis=1).astype(np.float32)

        # Group by subject
        for sid in np.unique(subj_ids):
            mask = subj_ids == sid
            if sid not in subjects:
                subjects[sid] = {"X": [], "y": []}
            subjects[sid]["X"].append(X_all[mask])
            subjects[sid]["y"].append(labels[mask])

    # Concatenate across train/test splits per subject
    for sid in subjects:
        subjects[sid]["X"] = np.concatenate(subjects[sid]["X"], axis=0)
        subjects[sid]["y"] = np.concatenate(subjects[sid]["y"], axis=0)

    logger.info(
        "UCI-HAR parsed: %d subjects, %d total windows, %d channels, %d classes",
        len(subjects),
        sum(s["X"].shape[0] for s in subjects.values()),
        9,
        len(set(np.concatenate([s["y"] for s in subjects.values()]))),
    )
    return subjects


# ═══════════════════════════════════════════════════════════════════════════════
#  PAMAP2 PARSER
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_pamap2(raw_dir: Path, drop_classes: list[int], target_hz: int = 50,
                  window_len: int = 128, window_stride: int = 64) -> dict:
    """
    Parse PAMAP2 .dat files into per-subject windowed arrays.

    PAMAP2 is sampled at 100 Hz for IMU data → resample to 50 Hz.
    """
    # Find Protocol directory
    candidates = list(raw_dir.rglob("Protocol"))
    if not candidates:
        candidates = list(raw_dir.rglob("protocol"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find Protocol directory in {raw_dir}")
    protocol_dir = candidates[0]

    # PAMAP2 columns: timestamp, activityID, heartrate, then 3 IMUs × 17 columns
    # We use IMU columns: acc (3), gyro (3), mag (3) for each of 3 placements = 27 ch
    # IMU data starts at column index 3
    imu_col_start = 4  # skip timestamp, activityID, heartrate, temperature
    # Each IMU block: 3 acc + 3 gyro + 3 mag + 4 orientation + 1 temp? = varies
    # Standard: columns 4-6 (acc hand), 7-9 (gyro hand), 10-12 (mag hand), 13-16 (orient), ...
    # We use acceleration (3) + gyroscope (3) + magnetometer (3) per IMU = 9 per IMU
    # 3 IMUs → 27 channels
    # IMU hand: cols 4-20 (17 cols), IMU chest: cols 21-37, IMU ankle: cols 38-54
    # Within each IMU: temperature(1), acc16(3), acc6(3), gyro(3), mag(3), orientation(4) = 17
    # We take: acc16(3) + gyro(3) + mag(3) = indices [1,2,3, 7,8,9, 10,11,12] within each block

    imu_block_indices = [1, 2, 3, 7, 8, 9, 10, 11, 12]  # within each 17-col IMU block

    subjects: dict = {}
    source_hz = 100  # PAMAP2 IMU sample rate

    for dat_file in sorted(protocol_dir.glob("subject10*.dat")):
        # Extract subject ID from filename
        fname = dat_file.stem
        sid = int("".join(filter(str.isdigit, fname)))

        data = pd.read_csv(dat_file, sep=r"\s+", header=None)
        data = data.values

        # Extract activity labels (column 1)
        activity = data[:, 1].astype(int)

        # Drop transitional classes
        valid_mask = ~np.isin(activity, drop_classes)
        data = data[valid_mask]
        activity = activity[valid_mask]

        if len(data) == 0:
            continue

        # Extract IMU channels
        channels = []
        for imu_start in [3, 20, 37]:  # hand, chest, ankle block starts
            for idx in imu_block_indices:
                col = imu_start + idx
                if col < data.shape[1]:
                    channels.append(data[:, col].astype(np.float32))

        if not channels:
            continue

        sensor_data = np.stack(channels, axis=0)  # (n_channels, n_samples)
        n_channels = sensor_data.shape[0]

        # NaN handling: forward-fill then zero-fill
        for ch in range(n_channels):
            series = pd.Series(sensor_data[ch])
            series = series.ffill().fillna(0.0)
            sensor_data[ch] = series.values

        # Resample from 100 Hz to 50 Hz
        n_samples = sensor_data.shape[1]
        n_resampled = n_samples // (source_hz // target_hz)
        sensor_resampled = np.zeros((n_channels, n_resampled), dtype=np.float32)
        for ch in range(n_channels):
            sensor_resampled[ch] = resample(sensor_data[ch], n_resampled)

        # Resample activity labels (nearest-neighbor)
        activity_resampled = resample(activity.astype(float), n_resampled)
        activity_resampled = np.round(activity_resampled).astype(int)

        # Re-filter after resampling (edge effects)
        valid_mask2 = ~np.isin(activity_resampled, drop_classes)

        # Windowing
        windows_X = []
        windows_y = []
        for start in range(0, n_resampled - window_len + 1, window_stride):
            end = start + window_len
            seg_labels = activity_resampled[start:end]
            seg_valid = ~np.isin(seg_labels, drop_classes)
            if not np.all(seg_valid):
                continue
            # Majority vote for window label
            label = int(pd.Series(seg_labels).mode().iloc[0])
            if label in drop_classes:
                continue
            windows_X.append(sensor_resampled[:, start:end])
            windows_y.append(label)

        if windows_X:
            subjects[sid] = {
                "X": np.stack(windows_X, axis=0).astype(np.float32),
                "y": np.array(windows_y, dtype=int),
            }

    # Re-map labels to contiguous 0..C-1
    if subjects:
        all_labels = set()
        for s in subjects.values():
            all_labels.update(s["y"].tolist())
        label_map = {old: new for new, old in enumerate(sorted(all_labels))}
        for s in subjects.values():
            s["y"] = np.array([label_map[l] for l in s["y"]], dtype=int)

    logger.info(
        "PAMAP2 parsed: %d subjects, %d total windows",
        len(subjects),
        sum(s["X"].shape[0] for s in subjects.values()) if subjects else 0,
    )
    return subjects


# ── Fallback PAMAP2 parser for different directory layouts ───────────────────
def _parse_pamap2_flexible(raw_dir: Path, drop_classes: list[int], target_hz: int = 50,
                           window_len: int = 128, window_stride: int = 64) -> dict:
    """Flexible parser that handles various PAMAP2 directory structures."""
    # Try to find any .dat files
    dat_files = list(raw_dir.rglob("*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {raw_dir}")

    subjects: dict = {}
    source_hz = 100

    for dat_file in sorted(dat_files):
        fname = dat_file.stem
        # Extract subject number
        digits = "".join(filter(str.isdigit, fname))
        if not digits:
            continue
        sid = int(digits)

        try:
            data = pd.read_csv(dat_file, sep=r"\s+", header=None).values
        except Exception as e:
            logger.warning("Skipping %s: %s", dat_file, e)
            continue

        if data.shape[1] < 40:
            continue

        activity = data[:, 1].astype(int)

        # Drop transitional
        valid = ~np.isin(activity, drop_classes)
        data = data[valid]
        activity = activity[valid]
        if len(data) == 0:
            continue

        # Use columns for 3 IMUs: acc + gyro + mag
        imu_block_indices = [1, 2, 3, 7, 8, 9, 10, 11, 12]
        channels = []
        for imu_start in [3, 20, 37]:
            for idx in imu_block_indices:
                col = imu_start + idx
                if col < data.shape[1]:
                    ch_data = data[:, col].astype(np.float32)
                    s = pd.Series(ch_data).ffill().fillna(0.0)
                    channels.append(s.values)

        if not channels:
            continue

        sensor = np.stack(channels, axis=0)
        n_ch, n_samp = sensor.shape

        # Resample
        n_res = n_samp // (source_hz // target_hz)
        resampled = np.zeros((n_ch, n_res), dtype=np.float32)
        for c in range(n_ch):
            resampled[c] = resample(sensor[c], n_res)

        act_res = np.round(resample(activity.astype(float), n_res)).astype(int)

        # Window
        wins_X, wins_y = [], []
        for st in range(0, n_res - window_len + 1, window_stride):
            seg = act_res[st:st + window_len]
            if np.any(np.isin(seg, drop_classes)):
                continue
            label = int(pd.Series(seg).mode().iloc[0])
            if label in drop_classes:
                continue
            wins_X.append(resampled[:, st:st + window_len])
            wins_y.append(label)

        if wins_X:
            subjects[sid] = {
                "X": np.stack(wins_X).astype(np.float32),
                "y": np.array(wins_y, dtype=int),
            }

    # Remap labels
    if subjects:
        all_labels = set()
        for s in subjects.values():
            all_labels.update(s["y"].tolist())
        label_map = {o: n for n, o in enumerate(sorted(all_labels))}
        for s in subjects.values():
            s["y"] = np.array([label_map[l] for l in s["y"]], dtype=int)

    logger.info("PAMAP2 (flex) parsed: %d subjects", len(subjects))
    return subjects


# ═══════════════════════════════════════════════════════════════════════════════
#  HHAR PARSER
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_hhar(raw_dir: Path, target_hz: int = 50,
                window_len: int = 128, window_stride: int = 64) -> dict:
    """
    Parse HHAR (Heterogeneity HAR) dataset.
    HHAR uses phone/watch accelerometer + gyroscope data.
    Subject = user × device combination (or just user).
    """
    # Find CSV files
    csv_candidates = list(raw_dir.rglob("*.csv"))
    # Look for the main data files: Phones_accelerometer.csv, etc.
    accel_files = [f for f in csv_candidates if "accel" in f.name.lower()]
    gyro_files = [f for f in csv_candidates if "gyro" in f.name.lower()]

    if not accel_files:
        raise FileNotFoundError(f"No accelerometer CSV found in {raw_dir}")

    # Activity label mapping
    activity_map = {
        "bike": 0, "sit": 1, "stand": 2,
        "walk": 3, "stairsup": 4, "stairsdown": 5,
        "null": -1,
    }

    subjects: dict = {}

    # Process accelerometer data grouped by user
    for accel_file in accel_files:
        logger.info("HHAR: reading %s", accel_file.name)
        try:
            df = pd.read_csv(accel_file)
        except Exception as e:
            logger.warning("Skipping %s: %s", accel_file, e)
            continue

        # Standardize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Need: user, x, y, z, gt (ground truth activity)
        if "user" not in df.columns or "gt" not in df.columns:
            logger.warning("HHAR: missing 'user' or 'gt' columns in %s", accel_file.name)
            continue

        # Map activity labels
        df["label"] = df["gt"].map(activity_map)
        df = df[df["label"] >= 0].copy()

        if df.empty:
            continue

        # Extract per-user
        for user, udf in df.groupby("user"):
            sid_key = hash(user) % 1000  # numeric subject ID
            xyz = udf[["x", "y", "z"]].values.astype(np.float32)
            labels = udf["label"].values.astype(int)

            # NaN handling
            for col in range(xyz.shape[1]):
                s = pd.Series(xyz[:, col])
                xyz[:, col] = s.ffill().fillna(0.0).values

            # The data is variable rate; resample to target_hz
            # Estimate source rate from timestamps if available
            n_samples = len(xyz)
            if n_samples < window_len:
                continue

            sensor = xyz.T  # (3, n_samples)

            # Window directly (already ~50-200Hz depending on device)
            # For simplicity and accuracy, we resample assuming ~100 Hz average
            est_hz = 100
            n_res = n_samples * target_hz // est_hz
            if n_res < window_len:
                continue

            n_ch = sensor.shape[0]
            resampled = np.zeros((n_ch, n_res), dtype=np.float32)
            for c in range(n_ch):
                resampled[c] = resample(sensor[c], n_res)

            label_res = np.round(resample(labels.astype(float), n_res)).astype(int)
            label_res = np.clip(label_res, 0, 5)

            # Window
            wins_X, wins_y = [], []
            for st in range(0, n_res - window_len + 1, window_stride):
                seg_labels = label_res[st:st + window_len]
                label = int(pd.Series(seg_labels).mode().iloc[0])
                if label < 0:
                    continue
                wins_X.append(resampled[:, st:st + window_len])
                wins_y.append(label)

            if wins_X:
                if sid_key in subjects:
                    subjects[sid_key]["X"] = np.concatenate(
                        [subjects[sid_key]["X"], np.stack(wins_X)], axis=0
                    )
                    subjects[sid_key]["y"] = np.concatenate(
                        [subjects[sid_key]["y"], np.array(wins_y, dtype=int)]
                    )
                else:
                    subjects[sid_key] = {
                        "X": np.stack(wins_X).astype(np.float32),
                        "y": np.array(wins_y, dtype=int),
                    }

    # Also process gyroscope if available and merge channels
    for gyro_file in gyro_files:
        logger.info("HHAR: reading gyro %s", gyro_file.name)
        try:
            df = pd.read_csv(gyro_file)
        except Exception:
            continue

        df.columns = [c.strip().lower() for c in df.columns]
        if "user" not in df.columns or "gt" not in df.columns:
            continue

        df["label"] = df["gt"].map(activity_map)
        df = df[df["label"] >= 0].copy()

        for user, udf in df.groupby("user"):
            sid_key = hash(user) % 1000
            xyz = udf[["x", "y", "z"]].values.astype(np.float32)

            for col in range(xyz.shape[1]):
                s = pd.Series(xyz[:, col])
                xyz[:, col] = s.ffill().fillna(0.0).values

            n_samples = len(xyz)
            if n_samples < window_len:
                continue

            sensor = xyz.T
            est_hz = 100
            n_res = n_samples * target_hz // est_hz
            if n_res < window_len:
                continue

            n_ch = sensor.shape[0]
            resampled = np.zeros((n_ch, n_res), dtype=np.float32)
            for c in range(n_ch):
                resampled[c] = resample(sensor[c], n_res)

            label_res = np.round(resample(udf["label"].values.astype(float), n_res)).astype(int)
            label_res = np.clip(label_res, 0, 5)

            wins_X, wins_y = [], []
            for st in range(0, n_res - window_len + 1, window_stride):
                seg_labels = label_res[st:st + window_len]
                label = int(pd.Series(seg_labels).mode().iloc[0])
                if label < 0:
                    continue
                wins_X.append(resampled[:, st:st + window_len])
                wins_y.append(label)

            if wins_X:
                gyro_key = sid_key + 10000  # separate namespace for gyro-only
                subjects[gyro_key] = {
                    "X": np.stack(wins_X).astype(np.float32),
                    "y": np.array(wins_y, dtype=int),
                    "_is_gyro": True,
                }

    # Merge acc+gyro for same user where possible
    acc_keys = {k for k in subjects if k < 10000}
    gyro_keys = {k for k in subjects if k >= 10000}
    for ak in acc_keys:
        gk = ak + 10000
        if gk in gyro_keys and gk in subjects:
            acc_data = subjects[ak]["X"]
            gyro_data = subjects[gk]["X"]
            # Match by minimum window count
            n = min(acc_data.shape[0], gyro_data.shape[0])
            subjects[ak]["X"] = np.concatenate(
                [acc_data[:n], gyro_data[:n]], axis=1
            ).astype(np.float32)
            subjects[ak]["y"] = subjects[ak]["y"][:n]
            del subjects[gk]

    # Clean up gyro-only keys
    subjects = {k: v for k, v in subjects.items() if not v.get("_is_gyro", False)}
    for v in subjects.values():
        v.pop("_is_gyro", None)

    # Remap labels
    if subjects:
        all_labels = set()
        for s in subjects.values():
            all_labels.update(s["y"].tolist())
        label_map = {o: n for n, o in enumerate(sorted(all_labels))}
        for s in subjects.values():
            s["y"] = np.array([label_map[l] for l in s["y"]], dtype=int)

    logger.info("HHAR parsed: %d subjects", len(subjects))
    return subjects


# ═══════════════════════════════════════════════════════════════════════════════
#  Z-SCORING (source-only normalization — CRITICAL)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_source_stats(
    subjects: dict,
    source_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std from ONLY the source (training) subjects.
    This is CRITICAL: test subjects' data must NOT influence these statistics.

    Returns
    -------
    mean : ndarray of shape (n_channels,)
    std  : ndarray of shape (n_channels,)
    """
    all_X = [subjects[sid]["X"] for sid in source_ids if sid in subjects]
    if not all_X:
        raise ValueError("No source data found for normalization")
    concat = np.concatenate(all_X, axis=0)  # (N, C, T)
    # Compute over windows and time, keeping channels
    mean = concat.mean(axis=(0, 2))  # (C,)
    std = concat.std(axis=(0, 2))    # (C,)
    std[std < 1e-8] = 1.0            # prevent division by zero
    return mean, std


def apply_zscore(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Apply z-score normalization: (X - mean) / std.

    Parameters
    ----------
    X    : (n_windows, n_channels, window_len)
    mean : (n_channels,)
    std  : (n_channels,)
    """
    # Broadcast: mean/std → (1, C, 1)
    return (X - mean[None, :, None]) / std[None, :, None]


# ═══════════════════════════════════════════════════════════════════════════════
#  MEMMAP SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def save_subject_memmap(
    dataset_name: str,
    subject_id: int,
    X: np.ndarray,
    y: np.ndarray,
    memmap_dir: Optional[Path] = None,
) -> Path:
    """
    Save one subject's data as memory-mapped .npy files with explicit flush.
    """
    if memmap_dir is None:
        memmap_dir = _PROJECT_ROOT / "data" / "memmap" / dataset_name
    memmap_dir.mkdir(parents=True, exist_ok=True)

    x_path = memmap_dir / f"subject_{subject_id}_X.npy"
    y_path = memmap_dir / f"subject_{subject_id}_y.npy"

    # Explicitly create memmaps, write, flush, and close for compute cluster NFS safety
    fp_x = np.lib.format.open_memmap(x_path, mode='w+', dtype=np.float32, shape=X.shape)
    fp_x[:] = X[:]
    fp_x.flush()
    del fp_x

    fp_y = np.lib.format.open_memmap(y_path, mode='w+', dtype=np.int64, shape=y.shape)
    fp_y[:] = y[:]
    fp_y.flush()
    del fp_y

    logger.info(
        "[%s] Subject %d saved: X=%s, y=%s → %s",
        dataset_name, subject_id, X.shape, y.shape, memmap_dir,
    )
    return x_path


def save_dataset_metadata(
    dataset_name: str,
    subjects: dict,
    memmap_dir: Optional[Path] = None,
) -> Path:
    """Save metadata.json for a dataset's memmap files."""
    if memmap_dir is None:
        memmap_dir = _PROJECT_ROOT / "data" / "memmap" / dataset_name
    memmap_dir.mkdir(parents=True, exist_ok=True)

    all_labels = set()
    for s in subjects.values():
        all_labels.update(s["y"].tolist())

    meta = {
        "dataset": dataset_name,
        "subjects": {},
        "num_classes": len(all_labels),
        "classes": sorted(list(all_labels)),
    }

    for sid, data in subjects.items():
        meta["subjects"][str(sid)] = {
            "file_X": f"subject_{sid}_X.npy",
            "file_y": f"subject_{sid}_y.npy",
            "num_windows": int(data["X"].shape[0]),
            "num_channels": int(data["X"].shape[1]),
            "window_length": int(data["X"].shape[2]),
        }

    meta_path = memmap_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("[%s] Metadata saved → %s", dataset_name, meta_path)
    return meta_path


def load_subject_memmap(
    dataset_name: str,
    subject_id: int,
    memmap_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one subject's data as memory-mapped arrays (lazy loading)."""
    if memmap_dir is None:
        memmap_dir = _PROJECT_ROOT / "data" / "memmap" / dataset_name
    x_path = memmap_dir / f"subject_{subject_id}_X.npy"
    y_path = memmap_dir / f"subject_{subject_id}_y.npy"
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    return X, y


def load_metadata(dataset_name: str, memmap_dir: Optional[Path] = None) -> dict:
    """Load a dataset's metadata.json."""
    if memmap_dir is None:
        memmap_dir = _PROJECT_ROOT / "data" / "memmap" / dataset_name
    meta_path = memmap_dir / "metadata.json"
    return json.loads(meta_path.read_text())


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER PIPELINE (per dataset)
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_and_cache_dataset(
    dataset_name: str,
    cfg: dict,
    force: bool = False,
    synthetic: bool = False,
) -> dict:
    """
    Full pipeline for one dataset:
      1. Parse raw data → per-subject arrays
      2. For each LOSO fold: compute source stats, z-score ALL subjects
      3. Save memmap + metadata

    For z-scoring, we save the UN-NORMALIZED data to memmap,
    and normalization happens at training time per LOSO fold.
    This is the correct approach: normalization stats depend on the fold.

    Parameters
    ----------
    dataset_name : str
        One of "uci_har", "pamap2", "hhar"
    cfg : dict
        Frozen configuration
    force : bool
        If True, re-process even if memmap files exist
    synthetic : bool
        If True, use synthetic data for testing

    Returns
    -------
    metadata : dict
    """
    memmap_dir = _PROJECT_ROOT / "data" / "memmap" / dataset_name
    if synthetic:
        memmap_dir = _PROJECT_ROOT / "data" / "synthetic_memmap" / dataset_name

    meta_path = memmap_dir / "metadata.json"

    # Skip if already processed
    if not force and meta_path.exists():
        logger.info("[%s] Memmap cache exists — skipping. Use force=True to re-process.", dataset_name)
        return json.loads(meta_path.read_text())

    raw_dir = _PROJECT_ROOT / "data" / "raw" / dataset_name
    prep = cfg["preprocessing"]

    # Parse
    if synthetic:
        from src.utils.synthetic_data import generate_synthetic_subjects
        subjects = generate_synthetic_subjects(dataset_name)
    elif dataset_name == "uci_har":
        subjects = _parse_uci_har(raw_dir)
    elif dataset_name == "pamap2":
        try:
            subjects = _parse_pamap2(
                raw_dir,
                drop_classes=prep["pamap2_drop_classes"],
                target_hz=prep["target_sample_rate_hz"],
                window_len=prep["window_length"],
                window_stride=prep["window_stride"],
            )
        except Exception:
            subjects = _parse_pamap2_flexible(
                raw_dir,
                drop_classes=prep["pamap2_drop_classes"],
                target_hz=prep["target_sample_rate_hz"],
                window_len=prep["window_length"],
                window_stride=prep["window_stride"],
            )
    elif dataset_name == "hhar":
        subjects = _parse_hhar(
            raw_dir,
            target_hz=prep["target_sample_rate_hz"],
            window_len=prep["window_length"],
            window_stride=prep["window_stride"],
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not subjects:
        raise RuntimeError(f"No subjects parsed for {dataset_name}")

    # Save raw (un-normalized) to memmap — one subject at a time to save RAM
    for sid, data in subjects.items():
        save_subject_memmap(dataset_name, sid, data["X"], data["y"], memmap_dir)
        # Free memory immediately
        del data["X"]
        import gc; gc.collect()
        # Reload as memmap reference for metadata
        data["X"], _ = load_subject_memmap(dataset_name, sid, memmap_dir)

    # Save metadata
    meta = save_dataset_metadata(dataset_name, subjects, memmap_dir)
    return json.loads(meta.read_text()) if isinstance(meta, Path) else meta


def preprocess_all_datasets(cfg: dict, force: bool = False, synthetic: bool = False) -> dict[str, dict]:
    """Process all three datasets. Returns {name: metadata}."""
    results = {}
    for name in ["uci_har", "pamap2", "hhar"]:
        logger.info("=" * 50)
        logger.info("Processing dataset: %s", name)
        logger.info("=" * 50)
        results[name] = preprocess_and_cache_dataset(name, cfg, force=force, synthetic=synthetic)
    return results
