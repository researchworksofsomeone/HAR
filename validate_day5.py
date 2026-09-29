import json
import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tinyhar_net import TinyHARNet
from src.energy.opcount import profile_inference_macs, ActionProfiler
from datetime import datetime

def run_day5_checkpoint():
    print("=" * 50)
    print("DAY 5 CHECKPOINT: Energy Accounting Module")
    print("=" * 50)
    
    # 1. Instantiate TinyHAR-Net
    model = TinyHARNet(in_channels=9, num_classes=6)
    
    # 2. Run profiler
    # batch_size=1, channels=9, length=128
    macs = profile_inference_macs(model, (1, 9, 128))
    
    target_macs = 1850000
    lower_bound = target_macs * 0.95
    upper_bound = target_macs * 1.05
    
    # 3. ASSERTION 1
    assert lower_bound <= macs <= upper_bound, f"ASSERTION 1 FAILED: {macs} not within 5% of {target_macs}"
    print(f"✓ Inference MACs: {macs:,} (Target: {target_macs:,}, Error: {abs(macs - target_macs)/target_macs * 100:.2f}%)")
    
    # 4. ASSERTION 2
    tent_ops = ActionProfiler.a3_tent_k(macs, k=2, B=8)
    assert tent_ops == 88800000, f"ASSERTION 2 FAILED: a3_TENT_k ops = {tent_ops}, expected 88,800,000"
    print(f"✓ a3_TENT_k (k=2, B=8) Ops: {tent_ops:,}")
    
    # 5. Load profiles.yaml
    profiles_path = PROJECT_ROOT / "configs" / "profiles.yaml"
    with open(profiles_path, "r") as f:
        profiles = yaml.safe_load(f)
        
    for p in ["P1", "P2", "P3", "P4"]:
        assert p in profiles, f"Profile {p} missing"
        assert "energy_per_mac_mj" in profiles[p], f"{p} missing energy_per_mac_mj"
        assert "sensitivity_multiplier" in profiles[p], f"{p} missing sensitivity_multiplier"
        
    print(f"✓ 4 Device Profiles validated from profiles.yaml")
    
    # 6. Save checkpoint
    ckpt = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inference_macs_verified": macs,
        "tent_ops_verified": tent_ops,
        "profiles_verified": True,
        "gate_passed": True
    }
    
    ckpt_path = PROJECT_ROOT / "checkpoints" / "checkpoint_day_5.json"
    ckpt_path.parent.mkdir(exist_ok=True)
    ckpt_path.write_text(json.dumps(ckpt, indent=2))
    
    print("\n[VALIDATION COMPLETE] checkpoint_day_5.json:")
    print(json.dumps(ckpt, indent=2))

if __name__ == "__main__":
    run_day5_checkpoint()
