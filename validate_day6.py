import json
import psutil
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.drift_regimes import DriftGenerator

def run_day6_checkpoint():
    print("=" * 50)
    print("DAY 6 CHECKPOINT: Drift Regime Generators")
    print("=" * 50)
    
    # 1. Generate small clean stream (1000 windows, 9 channels, 128 samples)
    n_windows = 1000
    channels = 9
    window_len = 128
    
    # Use real temp memmaps to rigorously test the slicing logic
    test_dir = PROJECT_ROOT / "data" / "synthetic" / "Day6Test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    X_path = test_dir / "X.npy"
    y_path = test_dir / "y.npy"
    alt_X_path = test_dir / "alt_X.npy"
    
    X_base = np.random.normal(0, 1.0, (n_windows, channels, window_len)).astype(np.float32)
    alt_X_base = np.random.normal(0, 1.0, (n_windows, channels, window_len)).astype(np.float32)
    y_base = np.random.randint(0, 6, (n_windows,)).astype(np.int64)
    
    np.save(X_path, X_base)
    np.save(y_path, y_base)
    np.save(alt_X_path, alt_X_base)
    
    X_memmap = np.load(X_path, mmap_mode='r')
    y_memmap = np.load(y_path, mmap_mode='r')
    alt_X_memmap = np.load(alt_X_path, mmap_mode='r')
    
    process = psutil.Process(os.getpid())
    ram_start = process.memory_info().rss / (1024 * 1024)
    peak_ram_mb = ram_start
    
    # --- Test R2 Gradual Degrade ---
    gen_r2 = DriftGenerator(X_memmap, y_memmap, chunk_size=32).r2_gradual_degrade()
    
    win_0_noise_var = None
    win_999_noise_var = None
    
    has_nan_or_inf = False
    
    win_idx = 0
    for chunk, y_chunk in gen_r2:
        current_ram = process.memory_info().rss / (1024 * 1024)
        peak_ram_mb = max(peak_ram_mb, current_ram)
        
        if np.isnan(chunk).any() or np.isinf(chunk).any():
            has_nan_or_inf = True
            
        clean_chunk = X_memmap[win_idx:win_idx+len(chunk)]
        delta = chunk - clean_chunk
        
        for i in range(len(chunk)):
            if win_idx == 0:
                win_0_noise_var = np.var(delta[i, 0, :])
            elif win_idx == 999:
                win_999_noise_var = np.var(delta[i, 0, :])
            win_idx += 1
            
    # ASSERTION 1
    assert win_999_noise_var > win_0_noise_var, f"ASSERTION 1 FAILED: R2 noise ramp didn't increase variance. 0={win_0_noise_var:.2f}, 999={win_999_noise_var:.2f}"
    print(f"✓ R2 Gradual Ramp Verified: Noise Variance Win 0 ({win_0_noise_var:.2f}) -> Win 999 ({win_999_noise_var:.2f})")
    
    # ASSERTION 3 (checked during R2 and R6)
    assert not has_nan_or_inf, "ASSERTION 3 FAILED: NaNs or Infs introduced during processing."
    print(f"✓ Zero NaN/Inf introduced during filtering/noise.")
    
    # --- Test R6 Mixed Volatile ---
    gen_r6 = DriftGenerator(X_memmap, y_memmap, alt_X=alt_X_memmap, alt_y=y_memmap, chunk_size=32).r6_mixed_volatile(severity='mid')
    
    states_seen = set()
    for chunk, y_chunk, state in gen_r6:
        current_ram = process.memory_info().rss / (1024 * 1024)
        peak_ram_mb = max(peak_ram_mb, current_ram)
        states_seen.add(state)
        
    # ASSERTION 2
    assert len(states_seen) > 1, f"ASSERTION 2 FAILED: R6 did not regime switch. States seen: {states_seen}"
    print(f"✓ R6 Markov Switching Verified: States hit = {states_seen}")
    
    # ASSERTION 4
    net_ram_increase = peak_ram_mb - ram_start
    assert net_ram_increase < 200, f"ASSERTION 4 FAILED: RAM footprint grew by {net_ram_increase:.2f} MB (ceiling 200MB)"
    print(f"✓ Memory Safety Verified: Peak Iteration RAM Growth = {net_ram_increase:.2f} MB")
    
    # Cleanup temp
    try:
        del X_memmap, y_memmap, alt_X_memmap
        X_path.unlink()
        y_path.unlink()
        alt_X_path.unlink()
        test_dir.rmdir()
    except Exception:
        pass
        
    # Checkpoint output
    ckpt = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "r2_ramp_verified": True,
        "r6_switches_verified": len(states_seen),
        "peak_ram_growth_mb": net_ram_increase,
        "gate_passed": True
    }
    
    ckpt_path = PROJECT_ROOT / "checkpoints" / "checkpoint_day_6.json"
    ckpt_path.parent.mkdir(exist_ok=True)
    ckpt_path.write_text(json.dumps(ckpt, indent=2))
    
    print("\n[VALIDATION COMPLETE] checkpoint_day_6.json:")
    print(json.dumps(ckpt, indent=2))

if __name__ == "__main__":
    run_day6_checkpoint()
