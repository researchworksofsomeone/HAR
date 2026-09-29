import gc
import os
import sys
import time
import psutil
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env_setup import init_environment, get_rss_gb
from src.controllers.data_pipeline import preprocess_all_datasets, load_subject_memmap, load_metadata
from src.models.tinyhar_net import build_tinyhar_net
from src.utils.synthetic_data import generate_synthetic_subjects

def run_memory_leak_test():
    print("=" * 60)
    print("PRE-FLIGHT: Memory Leak & Watchdog Stress Test")
    print("=" * 60)
    
    # 1. Init environment (Sets thread caps, determinism, seeds)
    cfg, device, ckpt_mgr = init_environment(config_path=str(PROJECT_ROOT / "configs" / "frozen.yaml"))
    
    # 2. Generate Synthetic Data
    print("\n[Synthetic Data Generation]")
    print("Bypassing raw parse. Generating synthetic memmaps...")
    metadata = preprocess_all_datasets(cfg, force=True, synthetic=True)
    
    dataset_name = "pamap2" # Use PAMAP2 as it has highest channel count (13) in our synthetic specs
    ds_meta = metadata[dataset_name]
    first_sid = list(ds_meta["subjects"].keys())[0]
    n_ch = ds_meta["subjects"][first_sid]["num_channels"]
    n_cls = ds_meta["num_classes"]
    
    # 3. Build Model
    model = build_tinyhar_net(cfg, n_ch, n_cls).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Stress Test Loop
    print(f"\n[Stress Test] Running 50 iterations on dataset: {dataset_name}, subject: {first_sid}")
    print(f"Channels: {n_ch}, Classes: {n_cls}")
    
    peak_rss = 0.0
    rss_history = []
    
    # Pre-allocate largest possible synthetic batch (e.g., all windows of a subject)
    # The synthetic subject has 50 windows. Let's make it bigger for the stress test
    # by just duplicating it to simulate a massive batch.
    X_mmap, y_mmap = load_subject_memmap(dataset_name, int(first_sid), 
                                         memmap_dir=PROJECT_ROOT / "data" / "synthetic_memmap" / dataset_name)
    
    X_large = np.tile(X_mmap, (100, 1, 1)) # (5000, 13, 128)
    y_large = np.tile(y_mmap, 100)         # (5000,)
    
    for i in range(50):
        # Memory tracking before loop
        gc.collect()
        rss_start = get_rss_gb()
        
        # Load batch into tensors
        xb = torch.from_numpy(X_large).to(device)
        yb = torch.from_numpy(y_large).to(device)
        
        # Forward
        logits = model(xb)
        loss = criterion(logits, yb)
        
        # Backward mock (just to build graph and release)
        loss.backward()
        
        # Delete tensors
        del xb, yb, logits, loss
        
        # Force GC
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Memory tracking after loop
        rss_end = get_rss_gb()
        peak_rss = max(peak_rss, rss_end)
        rss_history.append(rss_end)
        
        if (i + 1) % 10 == 0:
            print(f"  Iteration {i+1:2d}/50 | RSS: {rss_end:.3f} GB | Peak: {peak_rss:.3f} GB")
    
    # Analysis
    growth = rss_history[-1] - rss_history[0]
    print("\n[Results]")
    print(f"Initial RSS: {rss_history[0]:.3f} GB")
    print(f"Final RSS:   {rss_history[-1]:.3f} GB")
    print(f"Peak RSS:    {peak_rss:.3f} GB")
    print(f"Growth:      {growth:.3f} GB")
    
    # Assertions
    assert peak_rss < 3.5, f"OOM Danger! Peak RSS {peak_rss:.3f} GB exceeded 3.5 GB ceiling."
    assert growth < 0.05, f"Memory Leak Detected! RSS grew by {growth:.3f} GB over 50 iterations."
    print("✅ Memory Stress Test Passed. No leaks detected. Ceiling respected.")

if __name__ == "__main__":
    run_memory_leak_test()
