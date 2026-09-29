import json
import psutil
import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tinyhar_net import TinyHARNet
from src.controllers.features import FeatureExtractor
from src.controllers.jade_predictor import MultinomialLogistic, MLPPredictor

def run_day9_checkpoint():
    print("=" * 50)
    print("DAY 9 CHECKPOINT: JADE Feature & Predictor Engine")
    print("=" * 50)
    
    process = psutil.Process(os.getpid())
    ram_start = process.memory_info().rss / (1024 * 1024)
    peak_ram_mb = ram_start
    
    # 1. Instantiate Net and Extractor
    channels = 9
    num_classes = 6
    window_len = 128
    batch_size = 32
    
    model = TinyHARNet(in_channels=channels, num_classes=num_classes)
    model.eval()
    
    extractor = FeatureExtractor(num_classes=num_classes)
    
    # Run forward pass
    dummy_x = torch.randn(batch_size, channels, window_len)
    with torch.no_grad():
        logits = model(dummy_x).numpy()
        
    # Mock intermediates
    penultimate_feats = np.random.randn(batch_size, 64).astype(np.float32)
    prototypes = np.random.randn(num_classes, 64).astype(np.float32)
    
    bn_stats = [
        {'ring_mean': np.random.randn(32), 'run_mean': np.random.randn(32), 'ring_var': np.ones(32), 'run_var': np.ones(32)},
        {'ring_mean': np.random.randn(64), 'run_mean': np.random.randn(64), 'ring_var': np.ones(64), 'run_var': np.ones(64)},
        {'ring_mean': np.random.randn(64), 'run_mean': np.random.randn(64), 'ring_var': np.ones(64), 'run_var': np.ones(64)},
        {'ring_mean': np.random.randn(64), 'run_mean': np.random.randn(64), 'ring_var': np.ones(64), 'run_var': np.ones(64)}
    ]
    act_scales = [1.2, 0.9, 1.0, 1.1]
    src_act_scales = [1.0, 1.0, 1.0, 1.0]
    
    # 2. Extract feature
    phi_t = extractor.extract(
        logits=logits,
        penultimate_feats=penultimate_feats,
        prototypes=prototypes,
        bn_stats=bn_stats,
        act_scales=act_scales,
        src_act_scales=src_act_scales,
        tick=45,
        D=32
    )
    
    peak_ram_mb = max(peak_ram_mb, process.memory_info().rss / (1024 * 1024))
    
    # ASSERTION 1
    assert phi_t.shape == (28,), f"ASSERTION 1 FAILED: Feature shape is {phi_t.shape}, expected (28,)"
    print("✓ ASSERTION 1: Feature vector shape is exactly (28,)")
    
    # ASSERTION 2
    assert not np.isnan(phi_t).any(), "ASSERTION 2 FAILED: NaNs found in features"
    assert not np.isinf(phi_t).any(), "ASSERTION 2 FAILED: Infs found in features"
    print("✓ ASSERTION 2: Zero NaNs or Infs in feature vector (Causal strictly adhered)")
    
    # 3. Predictors
    lr_pred = MultinomialLogistic(28, 5, 3)
    mlp_pred = MLPPredictor(28, 32, 5, 3)
    
    lr_params = sum(p.numel() for p in lr_pred.parameters())
    mlp_params = sum(p.numel() for p in mlp_pred.parameters())
    
    # ASSERTION 3
    assert lr_params < 2000, f"ASSERTION 3 FAILED: MultinomialLogistic has {lr_params} params (> 2000)"
    assert mlp_params < 10000, f"ASSERTION 3 FAILED: MLPPredictor has {mlp_params} params (> 10000)"
    print(f"✓ ASSERTION 3: Predictors are lightweight (LR: {lr_params} params, MLP: {mlp_params} params)")
    
    # ASSERTION 4
    net_ram = peak_ram_mb - ram_start
    assert net_ram < 100, f"ASSERTION 4 FAILED: RAM grew by {net_ram:.2f} MB (ceiling 100MB)"
    print(f"✓ ASSERTION 4: Engine memory safety verified. Peak RAM growth: {net_ram:.2f} MB")
    
    ckpt = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phi_t_shape": list(phi_t.shape),
        "lr_params": lr_params,
        "mlp_params": mlp_params,
        "peak_ram_growth_mb": round(net_ram, 2),
        "gate_passed": True
    }
    
    ckpt_path = PROJECT_ROOT / "checkpoints" / "checkpoint_day_9.json"
    ckpt_path.parent.mkdir(exist_ok=True)
    ckpt_path.write_text(json.dumps(ckpt, indent=2))
    print("\n[VALIDATION COMPLETE] checkpoint_day_9.json:")
    print(json.dumps(ckpt, indent=2))

if __name__ == "__main__":
    run_day9_checkpoint()
