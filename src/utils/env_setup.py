"""
PerconAI — Environment Setup & Memory Guards
=============================================
This module MUST be imported at the very top of every notebook cell and script.
It enforces:
  • Deterministic behaviour (seeds, deterministic algorithms)
  • Thread pinning (OMP / Torch)
  • RAM ceiling (3.5 GB RSS hard limit)
  • GPU auto-detection for compute cluster (RTX 6000 Ada Lovelace)

NOTE (marimo / Jupyter):
PyTorch's thread pools are process-global and can only be configured ONCE.
In reactive notebooks a cell re-executes many times, so `pin_threads()` and
`init_environment()` are both guarded to be idempotent.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
import yaml

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PerconAI.env")

# ── Constants ────────────────────────────────────────────────────────────────
_MAX_RSS_GB = 28.0
_THREAD_COUNT = 4
_SEED = 42
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/utils/ → root

# ── Process-global idempotency guards ────────────────────────────────────────
_THREADS_PINNED = False
_INITIALIZED = False


# ═══════════════════════════════════════════════════════════════════════════════
#  1) SEED EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════
def seed_everything(seed: int = _SEED) -> None:
    """Pin all RNG sources for full reproducibility. Safe to call repeatedly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Seeds pinned → %d", seed)


# ═══════════════════════════════════════════════════════════════════════════════
#  2) THREAD PINNING (IDEMPOTENT)
# ═══════════════════════════════════════════════════════════════════════════════
def pin_threads(n: int = _THREAD_COUNT) -> None:
    """
    Force deterministic thread counts for OMP and Torch.

    Order matters:
      1. Env vars (always safe; needed by child processes / BLAS backends).
      2. torch.set_num_interop_threads(1)  ← MUST run first, once per process.
      3. torch.set_num_threads(n)          ← intra-op pool.

    Both torch calls are process-global and can only be applied once.
    The `_THREADS_PINNED` guard makes this safe to call from every cell.
    """
    global _THREADS_PINNED

    # 1) Env vars — cheap, reversible, safe on every call.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(n)

    # 2) Torch thread pools — only ever attempt once per process.
    if _THREADS_PINNED:
        logger.debug("Torch threads already pinned; skipping torch.set_* calls")
        return

    # Interop FIRST — this is the call that was crashing marimo.
    try:
        torch.set_num_interop_threads(1)
    except (RuntimeError, ValueError) as e:
        # Common case: parallel work already started in this process.
        logger.debug("torch.set_num_interop_threads skipped: %s", e)
    except Exception as e:
        logger.debug("torch.set_num_interop_threads unexpected error: %s", e)

    # Then intra-op pool.
    try:
        torch.set_num_threads(n)
    except (RuntimeError, ValueError) as e:
        logger.debug("torch.set_num_threads skipped: %s", e)
    except Exception as e:
        logger.debug("torch.set_num_threads unexpected error: %s", e)

    _THREADS_PINNED = True
    logger.info("Threads pinned → OMP=%s (torch pools attempted once)", n)


# ═══════════════════════════════════════════════════════════════════════════════
#  3) DETERMINISTIC ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════
def force_deterministic() -> None:
    """Enable Torch deterministic mode with full cuBLAS workspace."""
    # cuBLAS workspace env var must be set before the first cuBLAS call.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Deterministic algorithms enforced")


# ═══════════════════════════════════════════════════════════════════════════════
#  4) DEVICE DETECTION (compute cluster / GPU / CPU)
# ═══════════════════════════════════════════════════════════════════════════════
def get_device() -> torch.device:
    """Auto-detect best available device (compute cluster RTX 6000 Ada ⇒ cuda:0)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        name = torch.cuda.get_device_name(dev)
        # NOTE: attribute is `total_memory` (not `total_mem`).
        mem_gb = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        logger.info("GPU detected: %s (%.1f GB VRAM)", name, mem_gb)
    else:
        dev = torch.device("cpu")
        logger.info("No GPU — using CPU")
    return dev


# ═══════════════════════════════════════════════════════════════════════════════
#  5) RAM GUARD
# ═══════════════════════════════════════════════════════════════════════════════
class MemoryWatchdog:
    """Tracks RSS memory and raises MemoryError if it exceeds the ceiling."""

    def __init__(self, limit_gb: float = _MAX_RSS_GB):
        self.limit_gb = limit_gb
        self.process = psutil.Process(os.getpid())

    def get_rss_gb(self) -> float:
        return self.process.memory_info().rss / (1024 ** 3)

    def check(self) -> float:
        rss = self.get_rss_gb()
        if rss > self.limit_gb:
            gc.collect()
            rss = self.get_rss_gb()
            if rss > self.limit_gb:
                raise MemoryError(
                    f"MemoryWatchdog triggered! RSS: {rss:.2f} GB exceeds "
                    f"limit: {self.limit_gb:.2f} GB. Aborting to protect "
                    f"system stability."
                )
        return rss


def get_rss_gb() -> float:
    return MemoryWatchdog().get_rss_gb()


def check_ram_usage(max_gb: float = _MAX_RSS_GB) -> float:
    return MemoryWatchdog(max_gb).check()


# ═══════════════════════════════════════════════════════════════════════════════
#  6) CONFIG LOADER (with integrity check)
# ═══════════════════════════════════════════════════════════════════════════════
def load_frozen_config(path: Optional[str] = None) -> dict:
    """
    Load and validate the frozen YAML configuration.
    Raises KeyError if any required section is missing.
    """
    if path is None:
        path = str(_PROJECT_ROOT / "configs" / "frozen.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    required_keys = [
        "stream", "actions", "utility", "binning",
        "device_profiles", "random_seed", "training",
        "model", "preprocessing", "memory", "paths",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"Frozen config missing required keys: {missing}")

    for k in ("D", "H", "W"):
        assert k in cfg["stream"], f"stream.{k} missing"
    for k in ("k", "B", "N", "m"):
        assert k in cfg["actions"], f"actions.{k} missing"
    assert "lambda_grid" in cfg["utility"], "utility.lambda_grid missing"
    assert "harm_threshold" in cfg["binning"], "binning.harm_threshold missing"
    assert "help_threshold" in cfg["binning"], "binning.help_threshold missing"
    for p in ("P1", "P2", "P3", "P4"):
        assert p in cfg["device_profiles"], f"device_profiles.{p} missing"

    logger.info("Frozen config loaded and validated (%d top-level keys)", len(cfg))
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  7) CHECKPOINT MANAGER (resumable execution)
# ═══════════════════════════════════════════════════════════════════════════════
class CheckpointManager:
    """
    Manages checkpoint persistence for resumable multi-day execution.
    Optionally pushes checkpoints to a HuggingFace dataset repo.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_repo_id: Optional[str] = None,
    ):
        if checkpoint_dir is None:
            self.dir = _PROJECT_ROOT / "checkpoints"
        else:
            self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.hf_token = hf_token
        self.hf_repo_id = hf_repo_id
        self._hf_api = None

        if hf_token and hf_repo_id:
            try:
                from huggingface_hub import HfApi
                self._hf_api = HfApi(token=hf_token)
                self._hf_api.create_repo(
                    repo_id=hf_repo_id,
                    repo_type="dataset",
                    exist_ok=True,
                    private=False,
                )
                logger.info("CheckpointManager → %s + HF:%s", self.dir, hf_repo_id)
            except Exception as e:
                logger.warning("HF Hub init failed (will save locally only): %s", e)
                self._hf_api = None
        else:
            logger.info("CheckpointManager → %s (local only)", self.dir)

    @staticmethod
    def _config_hash() -> str:
        cfg_path = _PROJECT_ROOT / "configs" / "frozen.yaml"
        if cfg_path.exists():
            return hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16]
        return "no-config"

    def _upload_to_hf(self, local_path: Path, path_in_repo: str) -> None:
        """Upload a file to the HF dataset repo. Non-fatal on error."""
        if self._hf_api is None:
            return
        try:
            self._hf_api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                commit_message=f"Checkpoint: {path_in_repo}",
            )
            logger.info("  ↑ Uploaded to HF: %s/%s", self.hf_repo_id, path_in_repo)
        except Exception as e:
            logger.warning("  ↑ HF upload failed (checkpoint saved locally): %s", e)

    def save(
        self,
        day: int,
        gate_passed: bool,
        metrics: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> Path:
        """Save a day checkpoint locally and push to HF. Returns local path."""
        payload = {
            "day": day,
            "gate_passed": gate_passed,
            "metrics": metrics or {},
            "config_hash": self._config_hash(),
            "timestamp": datetime.now().isoformat(),
            "rss_gb": round(get_rss_gb(), 4),
        }
        if extra:
            payload.update(extra)
        out = self.dir / f"checkpoint_day_{day}.json"
        out.write_text(json.dumps(payload, indent=2, default=str))
        logger.info(
            "Checkpoint Day %d saved → gate_passed=%s  (%s)",
            day, gate_passed, out.name,
        )
        self._upload_to_hf(out, f"checkpoints/checkpoint_day_{day}.json")
        return out

    def upload_file(self, local_path: Path, path_in_repo: str) -> None:
        """Upload any arbitrary file to the HF repo (model weights, logs, ...)."""
        self._upload_to_hf(Path(local_path), path_in_repo)

    def load(self, day: int) -> Optional[dict]:
        """Load a previously saved checkpoint (or None if missing)."""
        path = self.dir / f"checkpoint_day_{day}.json"
        if path.exists():
            data = json.loads(path.read_text())
            logger.info(
                "Checkpoint Day %d loaded → gate_passed=%s",
                day, data["gate_passed"],
            )
            return data
        return None

    def gate_passed(self, day: int) -> bool:
        """Check if a specific day's gate has been passed."""
        ckpt = self.load(day)
        return ckpt is not None and ckpt.get("gate_passed", False)

    def require_gate(self, day: int) -> None:
        """Raise RuntimeError if the specified day gate has NOT been passed."""
        if not self.gate_passed(day):
            raise RuntimeError(
                f"Day {day} gate has NOT been passed. Cannot proceed. "
                f"Run Day {day} first."
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  8) MASTER INIT (safe to call from every cell)
# ═══════════════════════════════════════════════════════════════════════════════
def init_environment(
    config_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_repo_id: Optional[str] = None,
) -> tuple:
    """
    Master initializer — call once at the top of every notebook cell.

    Idempotent: safe to invoke on every marimo re-execution. The expensive
    one-time work (CUDA smoke test, log banner) runs only on the first
    successful call in this process.

    Returns
    -------
    (cfg: dict, device: torch.device, ckpt_mgr: CheckpointManager)
    """
    global _INITIALIZED

    if not _INITIALIZED:
        logger.info("=" * 60)
        logger.info("PerconAI Environment Initialization")
        logger.info("=" * 60)

    # 1) Threads first — before any BLAS / parallel work happens.
    pin_threads()

    # 2) Config (needed for the seed).
    cfg = load_frozen_config(config_path)

    # 3) Seeds — safe to reapply.
    seed_everything(cfg["random_seed"])

    # 4) Determinism — safe to reassert.
    force_deterministic()

    # 5) Device — cheap to re-detect.
    device = get_device()

    # 6) One-time hardware / CUDA smoke test (skip on re-runs).
    if not _INITIALIZED and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "6000" in gpu_name:
            logger.info("Hardware Check: ✓ RTX 6000 Detected (%s)", gpu_name)
        else:
            logger.warning("Hardware Check: ⚠️ Not an RTX 6000 (%s)", gpu_name)

        try:
            x = torch.randn(100, 100, device="cuda")
            y = x @ x.T
            logger.info("CUDA Mechanics: ✓ Matrix multiplication succeeded")
            del x, y
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error("CUDA Mechanics: ❌ FAILED (%s)", e)

    # 7) RAM check every time (cheap, and we want fresh numbers).
    rss = check_ram_usage(cfg["memory"]["max_rss_gb"])
    logger.info(
        "Current RSS: %.3f GB (limit: %.1f GB)",
        rss, cfg["memory"]["max_rss_gb"],
    )

    # 8) Checkpoint manager.
    ckpt_mgr = CheckpointManager(hf_token=hf_token, hf_repo_id=hf_repo_id)

    if not _INITIALIZED:
        logger.info("Environment ready ✓")
        logger.info("=" * 60)
        _INITIALIZED = True

    return cfg, device, ckpt_mgr
