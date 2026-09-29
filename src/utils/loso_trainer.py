"""
PerconAI — LOSO Trainer
=======================
Leave-One-Subject-Out cross-validation trainer with:
  • AdamW optimizer (lr=1e-3, wd=1e-4)
  • Cosine annealing scheduler (60 epochs)
  • Early stopping (patience=10, monitor=source_val_loss)
  • GPU support (auto-detected)
  • Checkpoint saving per fold
  • Source-only z-score normalization (per fold)
  • Verbose logging with RAM monitoring
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("PerconAI.trainer")

# ── Project root ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
#  EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════════
class EarlyStopping:
    """Early-stopping monitor with patience."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None
        self.triggered = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check whether to stop.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING WITH SOURCE-ONLY NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def _load_loso_fold(
    dataset_name: str,
    test_subject_id: int,
    subject_ids: list[int],
    memmap_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data for one LOSO fold with source-only z-scoring.

    Returns
    -------
    X_train, y_train : training data (all source subjects)
    X_val, y_val     : validation split from source subjects (10%)
    X_test, y_test   : held-out test subject
    """
    from src.controllers.data_pipeline import (
        apply_zscore,
        compute_source_stats,
        load_subject_memmap,
    )

    if memmap_dir is None:
        memmap_dir = _PROJECT_ROOT / "data" / "memmap" / dataset_name

    source_ids = [s for s in subject_ids if s != test_subject_id]

    # Load source subjects
    source_X_list, source_y_list = [], []
    for sid in source_ids:
        X, y = load_subject_memmap(dataset_name, sid, memmap_dir)
        source_X_list.append(np.array(X))  # materialize from memmap
        source_y_list.append(np.array(y))

    source_X = np.concatenate(source_X_list, axis=0)
    source_y = np.concatenate(source_y_list, axis=0)
    del source_X_list, source_y_list
    gc.collect()

    # Compute source-only normalization stats
    mean = source_X.mean(axis=(0, 2))  # (C,)
    std = source_X.std(axis=(0, 2))    # (C,)
    std[std < 1e-8] = 1.0

    # Normalize source data
    source_X = (source_X - mean[None, :, None]) / std[None, :, None]

    # Split source into train/val (90/10)
    n = len(source_X)
    indices = np.random.permutation(n)
    val_size = max(1, int(0.1 * n))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train = source_X[train_idx]
    y_train = source_y[train_idx]
    X_val = source_X[val_idx]
    y_val = source_y[val_idx]
    del source_X, source_y
    gc.collect()

    # Load and normalize test subject with SOURCE stats
    X_test, y_test = load_subject_memmap(dataset_name, test_subject_id, memmap_dir)
    X_test = np.array(X_test).astype(np.float32)
    y_test = np.array(y_test).astype(np.int64)
    X_test = (X_test - mean[None, :, None]) / std[None, :, None]

    return (
        X_train.astype(np.float32), y_train.astype(np.int64),
        X_val.astype(np.float32), y_val.astype(np.int64),
        X_test.astype(np.float32), y_test.astype(np.int64),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLE-FOLD TRAINER
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    device: torch.device,
    fold_name: str = "",
) -> tuple[nn.Module, dict]:
    """
    Train model for one LOSO fold.

    Returns
    -------
    model : nn.Module (best checkpoint restored)
    history : dict with train_loss, val_loss, val_acc per epoch
    """
    tcfg = cfg["training"]
    epochs = tcfg["epochs"]
    batch_size = tcfg["batch_size"]
    lr = tcfg["lr"]
    wd = tcfg["weight_decay"]
    patience = tcfg["early_stopping_patience"]

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                          num_workers=0, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_loss /= max(len(val_dl), 1)
        val_acc = correct / max(total, 1) * 100

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "[%s] Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  val_acc=%.1f%%",
                fold_name, epoch, epochs, train_loss, val_loss, val_acc,
            )

        # ── Early stopping ───────────────────────────────────────────────
        if early_stop.step(val_loss, model):
            logger.info("[%s] Early stopping at epoch %d (patience=%d)", fold_name, epoch, patience)
            break

    # Restore best model
    if early_stop.best_model_state is not None:
        model.load_state_dict(early_stop.best_model_state)
        logger.info("[%s] Best model restored (val_loss=%.4f)", fold_name, early_stop.best_loss)

    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[float, np.ndarray]:
    """
    Evaluate model on given data.

    Returns
    -------
    accuracy : float (0-100)
    predictions : ndarray
    """
    model.eval()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    accuracy = correct / max(total, 1) * 100
    return accuracy, np.array(all_preds)


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL LOSO CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def run_loso_cv(
    dataset_name: str,
    cfg: dict,
    device: torch.device,
    resume_from_subject: Optional[int] = None,
) -> dict:
    """
    Run full Leave-One-Subject-Out cross-validation for one dataset.

    Supports resumption: if checkpoints exist for some subjects,
    those folds are skipped.

    Returns
    -------
    results : dict with per-subject accuracies and mean accuracy
    """
    from src.controllers.data_pipeline import load_metadata
    from src.models.tinyhar_net import build_tinyhar_net

    meta = load_metadata(dataset_name)
    subject_ids = sorted([int(s) for s in meta["subjects"].keys()])
    num_classes = meta["num_classes"]
    # Get number of channels from first subject
    first_sid = list(meta["subjects"].keys())[0]
    num_channels = meta["subjects"][first_sid]["num_channels"]

    ckpt_dir = _PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = {"dataset": dataset_name, "subjects": {}, "accuracies": []}

    # Load any existing per-fold results
    fold_results_path = ckpt_dir / f"{dataset_name}_loso_progress.json"
    if fold_results_path.exists():
        existing = json.loads(fold_results_path.read_text())
        results["subjects"] = existing.get("subjects", {})
        results["accuracies"] = existing.get("accuracies", [])
        logger.info("[%s] Resuming LOSO — %d folds already completed",
                    dataset_name, len(results["subjects"]))

    for test_sid in subject_ids:
        sid_str = str(test_sid)
        fold_name = f"{dataset_name}/S{test_sid}"

        # Skip if already computed
        if sid_str in results["subjects"]:
            logger.info("[%s] Fold already complete — accuracy=%.1f%% — skipping",
                        fold_name, results["subjects"][sid_str]["test_accuracy"])
            continue

        if resume_from_subject is not None and test_sid < resume_from_subject:
            continue

        logger.info("=" * 50)
        logger.info("[%s] Starting LOSO fold (test subject = %d)", fold_name, test_sid)
        logger.info("=" * 50)

        t0 = time.time()

        # Load data with source-only normalization
        X_train, y_train, X_val, y_val, X_test, y_test = _load_loso_fold(
            dataset_name, test_sid, subject_ids,
        )

        logger.info(
            "[%s] Data: train=%d, val=%d, test=%d (channels=%d, classes=%d)",
            fold_name, len(X_train), len(X_val), len(X_test), num_channels, num_classes,
        )

        # Build fresh model
        model = build_tinyhar_net(cfg, num_channels, num_classes)

        # Train
        model, history = train_one_fold(
            model, X_train, y_train, X_val, y_val,
            cfg, device, fold_name,
        )

        # Evaluate on test subject
        test_acc, test_preds = evaluate_model(model, X_test, y_test, device)

        # Save model weights
        model_path = ckpt_dir / f"{dataset_name}_subject_{test_sid}_source.pth"
        torch.save(model.state_dict(), model_path)

        elapsed = time.time() - t0

        fold_result = {
            "test_accuracy": round(test_acc, 2),
            "val_accuracy": round(max(history["val_acc"]), 2) if history["val_acc"] else 0,
            "epochs_trained": len(history["train_loss"]),
            "best_val_loss": round(min(history["val_loss"]), 4) if history["val_loss"] else 0,
            "elapsed_seconds": round(elapsed, 1),
            "model_path": str(model_path),
        }
        results["subjects"][sid_str] = fold_result
        results["accuracies"].append(test_acc)

        logger.info(
            "[%s] ✓ Test accuracy = %.1f%% (%.0fs)",
            fold_name, test_acc, elapsed,
        )

        # Save progress (for resumption)
        fold_results_path.write_text(json.dumps(results, indent=2))

        # Cleanup
        del model, X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Compute summary
    accs = [results["subjects"][str(s)]["test_accuracy"] for s in subject_ids
            if str(s) in results["subjects"]]
    results["mean_accuracy"] = round(np.mean(accs), 2) if accs else 0
    results["std_accuracy"] = round(np.std(accs), 2) if accs else 0
    results["num_subjects"] = len(accs)

    logger.info(
        "[%s] LOSO complete: mean=%.1f%% ± %.1f%% (%d subjects)",
        dataset_name, results["mean_accuracy"], results["std_accuracy"], results["num_subjects"],
    )

    # Save final results
    final_path = ckpt_dir / f"{dataset_name}_loso_results.json"
    final_path.write_text(json.dumps(results, indent=2))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SANITY GATE CHECK
# ═══════════════════════════════════════════════════════════════════════════════
def check_sanity_gate(results: dict[str, dict]) -> tuple[bool, dict]:
    """
    Check Day 3 sanity gate: are LOSO accuracies within expected ranges?

    Expected ranges (literature-consistent):
      UCI-HAR:  75% – 88%
      PAMAP2:   65% – 80%
      HHAR:     75% – 90%
    """
    gates = {
        "uci_har": (75.0, 88.0),
        "pamap2": (65.0, 80.0),
        "hhar": (75.0, 90.0),
    }

    diagnostics = {}
    all_passed = True

    for dataset_name, (lo, hi) in gates.items():
        if dataset_name not in results:
            diagnostics[dataset_name] = {
                "passed": False,
                "error": f"Dataset {dataset_name} not found in results",
            }
            all_passed = False
            continue

        acc = results[dataset_name].get("mean_accuracy", 0)
        passed = lo <= acc <= hi

        diagnostics[dataset_name] = {
            "mean_accuracy": acc,
            "expected_range": [lo, hi],
            "passed": passed,
        }

        if not passed:
            all_passed = False
            if acc < lo:
                diagnostics[dataset_name]["diagnostic"] = (
                    f"Accuracy {acc:.1f}% is BELOW expected {lo}%. "
                    "Check: (1) z-scoring uses source-only stats, "
                    "(2) class filtering for PAMAP2, "
                    "(3) model architecture matches spec, "
                    "(4) training hyperparameters match frozen.yaml"
                )
            else:
                diagnostics[dataset_name]["diagnostic"] = (
                    f"Accuracy {acc:.1f}% is ABOVE expected {hi}%. "
                    "Check for potential data leakage in normalization."
                )
            logger.warning("[SANITY GATE] %s FAILED: %s",
                          dataset_name, diagnostics[dataset_name]["diagnostic"])
        else:
            logger.info("[SANITY GATE] %s PASSED: %.1f%% ∈ [%.0f%%, %.0f%%]",
                       dataset_name, acc, lo, hi)

    return all_passed, diagnostics
