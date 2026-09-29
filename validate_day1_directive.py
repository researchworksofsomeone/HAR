import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np

# Make imports work from project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env_setup import load_frozen_config, MemoryWatchdog
import torch

def setup_directories():
    required_dirs = [
        "configs", "data/raw", "data/processed", "data/memmap", "data/synthetic",
        "src/models", "src/controllers", "src/energy", "src/utils",
        "notebooks", "checkpoints", "logs"
    ]
    for d in required_dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
    print("Directory structure validated.")

def generate_synthetic_data():
    synthetic_dir = PROJECT_ROOT / "data" / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "FakeHAR": {"channels": 9, "classes": 6},
        "FakePAMAP": {"channels": 13, "classes": 8},
        "FakeHHAR": {"channels": 6, "classes": 6}
    }
    
    windows = 50
    window_len = 128
    files_created = 0
    rng = np.random.RandomState(42)
    
    for ds_name, props in datasets.items():
        ds_dir = synthetic_dir / ds_name
        ds_dir.mkdir(exist_ok=True)
        
        meta = {"subjects": {}}
        
        for sid in range(1, 4):  # 3 dummy subjects
            # X data
            x_path = ds_dir / f"subject_{sid}_X.npy"
            X = rng.randn(windows, props["channels"], window_len).astype(np.float32)
            fp_x = np.lib.format.open_memmap(x_path, mode='w+', dtype=np.float32, shape=X.shape)
            fp_x[:] = X[:]
            fp_x.flush()
            del fp_x
            
            # y data
            y_path = ds_dir / f"subject_{sid}_y.npy"
            y = rng.randint(0, props["classes"], size=(windows,)).astype(np.int64)
            fp_y = np.lib.format.open_memmap(y_path, mode='w+', dtype=np.int64, shape=y.shape)
            fp_y[:] = y[:]
            fp_y.flush()
            del fp_y
            
            files_created += 2
            
            meta["subjects"][str(sid)] = {
                "file_X": x_path.name,
                "file_y": y_path.name,
                "num_windows": windows,
                "num_channels": props["channels"],
                "window_length": window_len
            }
        
        meta_path = ds_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        files_created += 1
        
    print(f"Synthetic data generated: {files_created} files created in data/synthetic/")
    return files_created

def run_validation():
    setup_directories()
    
    # 1. Load frozen.yaml and verify keys
    cfg_path = PROJECT_ROOT / "configs" / "frozen.yaml"
    cfg = load_frozen_config(str(cfg_path))
    
    config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    
    # 2. Init MemoryWatchdog & dummy tensor operation
    watchdog = MemoryWatchdog(limit_gb=4.0)
    rss_before = watchdog.check()
    
    # CPU tensor op
    dummy = torch.randn(5000, 5000)
    _ = dummy @ dummy.T
    del dummy
    
    rss_after = watchdog.check()
    peak_ram_mb = max(rss_before, rss_after) * 1024
    print(f"MemoryWatchdog checked. Peak RAM: {peak_ram_mb:.2f} MB")
    
    # 3. Generate synthetic memmaps
    files_created = generate_synthetic_data()
    
    # 4. Save checkpoint_day_1.json
    ckpt = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_hash": config_hash,
        "synthetic_files_created": files_created,
        "peak_ram_during_validation_mb": round(peak_ram_mb, 2),
        "gate_passed": True
    }
    
    ckpt_path = PROJECT_ROOT / "checkpoints" / "checkpoint_day_1.json"
    ckpt_path.write_text(json.dumps(ckpt, indent=2))
    
    print("\n[VALIDATION COMPLETE] checkpoint_day_1.json:")
    print(json.dumps(ckpt, indent=2))
    
if __name__ == "__main__":
    run_validation()
