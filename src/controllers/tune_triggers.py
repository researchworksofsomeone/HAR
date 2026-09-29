import os
import sys
import yaml
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.triggers import TriggerController

def tune_threshold(signal: np.ndarray, target_rate: float, trigger_fn_name: str, low: float, high: float, iterations: int = 15):
    """
    Binary search for threshold that achieves the target fire rate.
    """
    best_tau = (low + high) / 2
    best_diff = float('inf')
    
    for _ in range(iterations):
        mid = (low + high) / 2
        
        # Instantiate controller with mid as the threshold
        if trigger_fn_name == "entropy_trigger":
            ctrl = TriggerController({"tau_E": mid})
            fires = sum(1 for s in signal if ctrl.entropy_trigger(np.array([s]))[0])
        elif trigger_fn_name == "surprise_trigger":
            ctrl = TriggerController({"tau_S": mid})
            fires = sum(1 for s in signal if ctrl.surprise_trigger(np.array([s]))[0])
        elif trigger_fn_name == "shift_trigger":
            ctrl = TriggerController({"adwin_threshold": mid})
            fires = sum(1 for s in signal if ctrl.shift_trigger(s)[0])
        elif trigger_fn_name == "ph_drift":
            ctrl = TriggerController({"ph_threshold": mid})
            fires = sum(1 for s in signal if ctrl.ph_drift(s)[0])
            
        rate = fires / len(signal)
        diff = abs(rate - target_rate)
        
        if diff < best_diff:
            best_diff = diff
            best_tau = mid
            
        if rate > target_rate:
            # Firing too much, need higher threshold (for most triggers where > fires)
            # Or lower threshold? 
            # Entropy/Surprise: > tau means fire. If rate > target, threshold is too low.
            low = mid
        else:
            high = mid
            
    return float(best_tau)

def tune_all_triggers(target_fire_rate: float = 0.15):
    print(f"Tuning proxy triggers for target fire rate: {target_fire_rate * 100:.1f}%")
    
    # 1. Mock proxy signals for the tuning phase (as actual JADE features are Day 9)
    # We assume a mix of clean and mild drift representing the controller-training set.
    np.random.seed(42)
    n_samples = 2000
    
    # Entropy proxy (typically 0.1 to 2.0)
    entropy_signal = np.random.normal(0.8, 0.4, n_samples)
    
    # Surprise proxy (typically 0.0 to 1.0 cosine distance)
    surprise_signal = np.random.normal(0.3, 0.15, n_samples)
    
    # BN displacement (typically 0.0 to 0.5)
    bn_signal = np.random.normal(0.1, 0.05, n_samples)
    
    # 2. Tune each
    tau_E = tune_threshold(entropy_signal, target_fire_rate, "entropy_trigger", 0.0, 3.0)
    tau_S = tune_threshold(surprise_signal, target_fire_rate, "surprise_trigger", 0.0, 1.0)
    
    # ADWIN and PH thresholds are trickier since they are stateful and look at change.
    # We will simulate a signal with injected jumps for them to catch.
    jump_signal = np.random.normal(0.5, 0.1, n_samples)
    # Add some random jumps
    for i in range(100, n_samples, 200):
        jump_signal[i:i+50] += 0.4
        
    tau_adwin = tune_threshold(jump_signal, target_fire_rate, "shift_trigger", 0.01, 1.0)
    tau_ph = tune_threshold(jump_signal, target_fire_rate, "ph_drift", 1.0, 20.0)
    
    tuned_dict = {
        "tau_E": round(tau_E, 4),
        "tau_S": round(tau_S, 4),
        "adwin_threshold": round(tau_adwin, 4),
        "ph_threshold": round(tau_ph, 4)
    }
    
    out_path = PROJECT_ROOT / "configs" / "tuned_triggers.yaml"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(tuned_dict, f)
        
    print(f"Tuning complete. Saved to configs/tuned_triggers.yaml:\n{tuned_dict}")
    return tuned_dict

if __name__ == "__main__":
    tune_all_triggers()
