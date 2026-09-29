#!/usr/bin/env python3
"""
PerconAI — Day 1 Validation
============================
Runs all Day 1 checks (directory structure, config validation, env init,
RAM baseline) and writes a resumable checkpoint via CheckpointManager.

Usage (script):
    python validate_day1.py

Usage (notebook / marimo):
    from validate_day1 import run_day1_checkpoint
    result = run_day1_checkpoint()
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

# ── Project root (directory containing this file; also the marimo cwd) ──
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ────────────────────────────────────────────────────────────────
REQUIRED_DIRS = [
    "configs", "data/raw", "data/processed", "data/memmap",
    "src/models", "src/controllers", "src/energy", "src/utils",
    "notebooks", "checkpoints",
]

REQUIRED_TOP_KEYS = [
    "stream", "actions", "utility", "binning",
    "device_profiles", "random_seed", "training",
    "model", "preprocessing", "memory", "paths",
]

# Day-1 RAM gate: smoke check that torch + CUDA context loaded without
# pathological baseline inflation. NOT the production ceiling (3.5 GB).
RAM_BASELINE_MB = 5000.0


def _require(cond: bool, msg: str) -> None:
    """Explicit check — survives `python -O` and produces a clear message."""
    if not cond:
        raise AssertionError(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  Task 1.1 — Directory structure
# ═══════════════════════════════════════════════════════════════════════════════
def _task_1_1_directories() -> bool:
    print("\n📁 Task 1.1 — Directory Structure")
    print("-" * 40)
    ok = True
    for d in REQUIRED_DIRS:
        full = PROJECT_ROOT / d
        full.mkdir(parents=True, exist_ok=True)
        exists = full.is_dir()
        ok = ok and exists
        print(f"  {'✓' if exists else '✗'} {d}/")
    print(f"\n{'✅ All directories verified' if ok else '❌ Missing directories'}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
#  Task 1.2 — Frozen config validation
# ═══════════════════════════════════════════════════════════════════════════════
def _task_1_2_config(config_path: Path) -> dict:
    print("\n📋 Task 1.2 — Frozen Configuration")
    print("-" * 40)

    _require(config_path.exists(), f"frozen.yaml not found at {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for key in REQUIRED_TOP_KEYS:
        _require(key in cfg, f"Missing top-level key: {key}")
        print(f"  ✓ {key}")

    # Concrete value assertions
    _require(cfg["stream"]["D"] == 32, "stream.D must be 32")
    _require(cfg["stream"]["H"] == 64, "stream.H must be 64")
    _require(cfg["stream"]["W"] == 64, "stream.W must be 64")
    _require(cfg["actions"]["k"] == 2, "actions.k must be 2")
    _require(cfg["actions"]["B"] == 8, "actions.B must be 8")
    _require(cfg["actions"]["N"] == 256, "actions.N must be 256")
    _require(cfg["actions"]["m"] == 0.5, "actions.m must be 0.5")
    _require(
        len(cfg["utility"]["lambda_grid"]) == 8,
        "utility.lambda_grid must have exactly 8 entries",
    )
    _require(cfg["binning"]["harm_threshold"] == -0.5,
             "binning.harm_threshold must be -0.5")
    _require(cfg["binning"]["help_threshold"] == 0.5,
             "binning.help_threshold must be 0.5")
    for p in ("P1", "P2", "P3", "P4"):
        _require(p in cfg["device_profiles"], f"device_profiles.{p} missing")
    _require(cfg["random_seed"] == 42, "random_seed must be 42")

    print("✅ All config values validated")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  Task 1.3 — Environment initialization + smoke stats
# ═══════════════════════════════════════════════════════════════════════════════
def _task_1_3_env(config_path: Path):
    print("\n🔧 Task 1.3 — Environment Initialization")
    print("-" * 40)

    from src.utils.env_setup import init_environment, get_rss_gb
    import torch

    cfg, device, ckpt_mgr = init_environment(str(config_path))

    rss_mb = get_rss_gb() * 1024
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  CUDA:           {torch.cuda.is_available()}")
    print(f"  Device:         {device}")
    print(f"  Threads:        {torch.get_num_threads()}")
    print(f"  Deterministic:  {torch.are_deterministic_algorithms_enabled()}")
    print(f"  RSS:            {get_rss_gb():.3f} GB ({rss_mb:.1f} MB)")

    return cfg, device, ckpt_mgr, rss_mb, torch


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════
def run_day1_checkpoint(config_path: Optional[str] = None) -> dict[str, Any]:
    """
    Run the Day 1 validation pipeline and persist a checkpoint.

    Idempotent — safe to call on every notebook re-execution. `init_environment`
    is itself idempotent, and the checkpoint file is simply overwritten.

    Returns
    -------
    dict with keys:
        gate_passed : bool
        checkpoint  : str  (path to checkpoint_day_1.json)
        checks      : dict (individual gate results)
        rss_mb      : float
    """
    print("=" * 60)
    print("  PerconAI — DAY 1 VALIDATION")
    print("=" * 60)

    cfg_path = (
        Path(config_path)
        if config_path
        else PROJECT_ROOT / "configs" / "frozen.yaml"
    )

    dirs_ok = _task_1_1_directories()
    _task_1_2_config(cfg_path)                    # raises on any failure
    cfg, device, ckpt_mgr, rss_mb, torch = _task_1_3_env(cfg_path)

    print("\n🚦 DAY 1 CHECKPOINT")
    print("-" * 40)
    checks = {
        # If we reached this line, Task 1.2 already raised on any problem.
        "config_valid": True,
        "ram_under_baseline": rss_mb < RAM_BASELINE_MB,
        "directories_exist": dirs_ok,
        "seed_set": cfg["random_seed"] == 42,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }
    for name, passed in checks.items():
        print(f"  {'✅ PASS' if passed else '❌ FAIL'} — {name}")
    print(f"\n  RAM: {rss_mb:.1f} MB "
          f"(Day-1 baseline: {RAM_BASELINE_MB:.0f} MB)")

    gate_passed = all(checks.values())
    ckpt_path = ckpt_mgr.save(
        day=1,
        gate_passed=gate_passed,
        metrics={"rss_mb": round(rss_mb, 2), **checks},
    )

    print(f"\n{'=' * 60}")
    print("  ✅ DAY 1 GATE PASSED" if gate_passed else "  ❌ DAY 1 GATE FAILED")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'=' * 60}")

    print("\ncheckpoint_day_1.json:")
    print(json.dumps(json.loads(ckpt_path.read_text()), indent=2))

    return {
        "gate_passed": gate_passed,
        "checkpoint": str(ckpt_path),
        "checks": checks,
        "rss_mb": round(rss_mb, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    result = run_day1_checkpoint()
    sys.exit(0 if result["gate_passed"] else 1)
