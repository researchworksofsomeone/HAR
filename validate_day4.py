import json
import torch
import copy
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tinyhar_net import TinyHARNet
from src.controllers.actions import (
    a0_NOOP, a1_BN_RECAL, a2_EM_PRIOR, a3_TENT_k, a4_RESET, 
    a12_LEAN_ALWAYS, a13_LAME_STYLE
)

def run_day4_checkpoint():
    print("=" * 50)
    print("DAY 4 CHECKPOINT: Action Menu Validation")
    print("=" * 50)
    
    # 1. Setup dummy model and data
    batch_size = 8
    channels = 13
    num_classes = 8
    model = TinyHARNet(in_channels=channels, num_classes=num_classes)
    model.eval()
    
    dummy_x = torch.randn(batch_size, channels, 128)
    theta_src = copy.deepcopy(model.state_dict())
    
    # Dummy buffers
    buffer_x = torch.randn(64, channels, 128)
    buffer_preds = torch.softmax(torch.randn(256, num_classes), dim=1)
    C_src = torch.eye(num_classes)
    buffer_stats = {}
    buffer_feats = torch.randn(batch_size, 64)
    
    # 2. Test Actions
    actions = [
        ("a0_NOOP", lambda m: a0_NOOP(m, dummy_x)),
        ("a1_BN_RECAL", lambda m: a1_BN_RECAL(m, buffer_x, m=0.5)),
        ("a2_EM_PRIOR", lambda m: a2_EM_PRIOR(m, buffer_preds, C_src)),
        ("a3_TENT_k", lambda m: a3_TENT_k(m, dummy_x, k=2)),  # Using dummy_x (B=8)
        ("a4_RESET", lambda m: a4_RESET(m, theta_src)),
        ("a12_LEAN_ALWAYS", lambda m: a12_LEAN_ALWAYS(m, dummy_x, buffer_stats)),
        ("a13_LAME_STYLE", lambda m: a13_LAME_STYLE(m, dummy_x, buffer_feats)),
    ]
    
    results = {}
    
    for name, act_fn in actions:
        # Clone model to prevent state leakage during tests
        m_clone = copy.deepcopy(model)
        
        updated_model, ops = act_fn(m_clone)
        
        # Test forward pass shape
        with torch.no_grad():
            updated_model.eval()
            out = updated_model(dummy_x)
        
        assert out.shape == (batch_size, num_classes), f"{name} shape mismatch: {out.shape}"
        if name not in ["a0_NOOP", "a4_RESET"]:
            assert ops > 0, f"{name} failed ops > 0 check: {ops}"
        elif name == "a0_NOOP":
            assert ops == 0, f"{name} ops must be 0"
            
        print(f"✓ {name.ljust(18)} | Ops: {ops} | Output shape: {out.shape}")
        results[name] = {"ops": ops, "shape_valid": True}
        
    # 3. Save Checkpoint
    ckpt = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "actions_tested": len(results),
        "results": results,
        "gate_passed": True
    }
    
    ckpt_path = PROJECT_ROOT / "checkpoints" / "checkpoint_day_4.json"
    ckpt_path.parent.mkdir(exist_ok=True)
    ckpt_path.write_text(json.dumps(ckpt, indent=2))
    
    print("\n[VALIDATION COMPLETE] checkpoint_day_4.json:")
    print(json.dumps(ckpt, indent=2))

if __name__ == "__main__":
    run_day4_checkpoint()
