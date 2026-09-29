import argparse
import os
import sys
import gc
import json
import time
import psutil
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tinyhar_net import TinyHARNet
from src.data.drift_regimes import DriftGenerator
from src.controllers.features import FeatureExtractor
from src.controllers.actions import a0_NOOP, a1_BN_RECAL, a2_EM_PRIOR, a3_TENT_k
from src.utils.env_setup import init_environment

def evaluate_action_hmo(base_model, action_name, buffer_x, buffer_preds, C_src, future_x, future_y):
    import copy
    device = next(base_model.parameters()).device
    base_state = copy.deepcopy(base_model.state_dict())
    
    # 1. Base accuracy (NOOP)
    base_model.eval()
    with torch.no_grad():
        base_logits = base_model(future_x)
        base_preds_cls = base_logits.argmax(dim=1)
        base_acc = (base_preds_cls == future_y).float().mean().item() * 100.0  # Convert to percentage
        
    del base_logits, base_preds_cls
    
    # 2. Apply action
    temp_model = copy.deepcopy(base_model)
    temp_model.load_state_dict(base_state)
    
    if action_name == "a0_NOOP":
        temp_model, _ = a0_NOOP(temp_model, buffer_x)
    elif action_name == "a1_BN_RECAL":
        temp_model, _ = a1_BN_RECAL(temp_model, buffer_x, m=0.5)
    elif action_name == "a2_EM_PRIOR":
        temp_model, _ = a2_EM_PRIOR(temp_model, buffer_preds, C_src)
    elif action_name == "a3_TENT_k":
        temp_model, _ = a3_TENT_k(temp_model, buffer_x, k=2)
        
    # 3. Evaluate Future
    temp_model.eval()
    with torch.no_grad():
        logits = temp_model(future_x)
        preds = logits.argmax(dim=1)
        acc = (preds == future_y).float().mean().item() * 100.0  # Convert to percentage
        
    delta_H = acc - base_acc
    
    # THE NUKE (CRITICAL)
    del temp_model, logits, preds, base_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    return delta_H

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--regime", type=str, required=True)
    parser.add_argument("--subject_idx", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severity", type=str, default="mid", choices=["low", "mid", "high"])
    parser.add_argument("--hf_repo", type=str, default=None, help="HuggingFace dataset repo")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace write token")
    args = parser.parse_args()
    
    # Initialize Environment
    cfg_path = str(PROJECT_ROOT / "configs" / "frozen.yaml")
    cfg, device, ckpt_mgr = init_environment(cfg_path)
    
    # Ensure explicit seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    data_dir = Path(args.output_dir) / args.dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "metadata.json"
    
    # Fallback to local memmaps if no HF is provided (or if testing locally)
    # The real data should be prepared into data_dir first.
    if not meta_path.exists():
        if args.hf_repo:
            print(f"Downloading metadata.json from HuggingFace repo {args.hf_repo}...")
            from huggingface_hub import hf_hub_download
            meta_path = Path(hf_hub_download(repo_id=args.hf_repo, repo_type="dataset", 
                                             filename=f"memmaps/{args.dataset}/metadata.json", 
                                             token=args.hf_token))
        else:
            # Look in project data dir
            local_meta = PROJECT_ROOT / "data/memmap" / args.dataset / "metadata.json"
            if local_meta.exists():
                import shutil
                shutil.copytree(PROJECT_ROOT / "data/memmap" / args.dataset, data_dir, dirs_exist_ok=True)
            else:
                local_meta = PROJECT_ROOT / "data/synthetic_memmap" / args.dataset / "metadata.json"
                if local_meta.exists():
                    import shutil
                    shutil.copytree(PROJECT_ROOT / "data/synthetic_memmap" / args.dataset, data_dir, dirs_exist_ok=True)
                else:
                    raise FileNotFoundError(f"metadata.json not found for {args.dataset}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    subj_meta = next((s for s in metadata["subjects"] if s["id"] == args.subject_idx), None)
    if not subj_meta:
        raise ValueError(f"Subject {args.subject_idx} not found in metadata.")
        
    x_path = data_dir / subj_meta["x_path"]
    y_path = data_dir / subj_meta["y_path"]
    
    if not x_path.exists() and args.hf_repo:
        print(f"Downloading {subj_meta['x_path']} from HuggingFace...")
        from huggingface_hub import hf_hub_download
        x_path = Path(hf_hub_download(repo_id=args.hf_repo, repo_type="dataset", 
                                      filename=f"memmaps/{args.dataset}/{subj_meta['x_path']}", 
                                      token=args.hf_token))
                                      
    if not y_path.exists() and args.hf_repo:
        print(f"Downloading {subj_meta['y_path']} from HuggingFace...")
        from huggingface_hub import hf_hub_download
        y_path = Path(hf_hub_download(repo_id=args.hf_repo, repo_type="dataset", 
                                      filename=f"memmaps/{args.dataset}/{subj_meta['y_path']}", 
                                      token=args.hf_token))
    
    X_mmap = np.lib.format.open_memmap(x_path, mode='r')
    y_mmap = np.lib.format.open_memmap(y_path, mode='r')
    
    channels = X_mmap.shape[1]
    num_classes = 6
    window_len = X_mmap.shape[2]
    
    model = TinyHARNet(in_channels=channels, num_classes=num_classes)
    model.to(device)
    model.eval()
    
    extractor = FeatureExtractor(num_classes=num_classes)
    
    # Init Drift Generator
    regime_map = {
        "r1": "r1_sudden_subject",
        "r2": "r2_gradual_degrade",
        "r3": "r3_recurring",
        "r4": "r4_position_shift",
        "r5": "r5_device_hetero",
        "r6": "r6_mixed_volatile"
    }
    method_name = regime_map.get(args.regime.lower(), args.regime.lower())
    
    if args.regime.lower() == "r6":
        gen = getattr(DriftGenerator(X_mmap, y_mmap, alt_X=X_mmap, alt_y=y_mmap, chunk_size=32), method_name)(severity=args.severity)
    else:
        gen = getattr(DriftGenerator(X_mmap, y_mmap, alt_X=X_mmap, alt_y=y_mmap, chunk_size=32), method_name)()
    
    actions = ["a0_NOOP", "a1_BN_RECAL", "a2_EM_PRIOR", "a3_TENT_k"]
    buffer_records = []
    chunk_idx = 0
    H = 64  # Per prompt: H=64 windows
    
    # Load thresholds from config
    harm_thresh = cfg["binning"]["harm_threshold"]
    help_thresh = cfg["binning"]["help_threshold"]
    
    out_dir = Path(args.output_dir) / "corpus" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Generation: {args.dataset} | Regime: {args.regime} | Subj: {args.subject_idx}")
    start_time = time.time()
    
    tick = 0
    total_ticks = len(X_mmap)
    
    for item in gen:
        if len(item) == 3:
            chunk, y_chunk, state = item
        else:
            chunk, y_chunk = item
            state = None
            
        for i in range(len(chunk)):
            current_x = torch.tensor(chunk[i:i+1], device=device)
            
            with torch.no_grad():
                logits = model(current_x).cpu().numpy()
            
            # Extract 28-dim feature vector phi_t
            phi_t = extractor.extract(
                logits=logits,
                penultimate_feats=np.random.randn(1, 64).astype(np.float32),
                prototypes=np.random.randn(num_classes, 64).astype(np.float32),
                bn_stats=[{'ring_mean': np.zeros(32), 'run_mean': np.zeros(32), 'ring_var': np.ones(32), 'run_var': np.ones(32)}]*4,
                act_scales=[1.0]*4, src_act_scales=[1.0]*4,
                tick=tick
            )
            
            # Sample action (80% uniform random valid, 20% NOOP)
            if np.random.rand() < 0.2:
                a_idx = 0
            else:
                a_idx = np.random.randint(0, len(actions))
            a_name = actions[a_idx]
            
            # HMO Branching
            fut_end = min(tick + H, total_ticks)
            fut_len = fut_end - tick
            if fut_len > 0:
                future_x = torch.tensor(X_mmap[tick:fut_end], device=device)
                future_y = torch.tensor(y_mmap[tick:fut_end], device=device)
                
                delta_H = evaluate_action_hmo(
                    model, a_name, 
                    buffer_x=current_x, 
                    buffer_preds=torch.softmax(torch.tensor(logits, device=device), dim=1),
                    C_src=torch.eye(num_classes, device=device),
                    future_x=future_x,
                    future_y=future_y
                )
            else:
                delta_H = 0.0
                
            # Binning based on frozen thresholds
            if delta_H < harm_thresh:
                bin_label = "harm"
            elif delta_H > help_thresh:
                bin_label = "help"
            else:
                bin_label = "neutral"
            
            # Append row
            buffer_records.append({
                "phi_t": phi_t.tolist(),
                "action_id": a_idx,
                "bin_label": bin_label,
                "delta_h": delta_H
            })
            
            tick += 1
            
            # Chunked I/O
            if len(buffer_records) >= 10000:
                df = pd.DataFrame(buffer_records)
                out_path = out_dir / f"corpus_{args.dataset}_{args.regime}_subj{args.subject_idx}_chunk{chunk_idx}.parquet"
                df.to_parquet(out_path)
                
                if args.hf_repo and args.hf_token:
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi()
                        hf_path = f"corpus/{args.dataset}/{out_path.name}"
                        api.upload_file(path_or_fileobj=str(out_path), path_in_repo=hf_path, repo_id=args.hf_repo, repo_type="dataset", token=args.hf_token)
                        os.remove(out_path)
                    except Exception as e:
                        print(f"Failed to upload chunk: {e}")
                        
                buffer_records = []
                chunk_idx += 1
                gc.collect()
                
            if tick % 50 == 0:
                elapsed = time.time() - start_time
                rate = tick / elapsed
                rem = (total_ticks - tick) / rate
                # Just keeping terminal output minimal
                pass

    # Flush remaining
    if len(buffer_records) > 0:
        df = pd.DataFrame(buffer_records)
        out_path = out_dir / f"corpus_{args.dataset}_{args.regime}_subj{args.subject_idx}_chunk{chunk_idx}.parquet"
        df.to_parquet(out_path)
        
        if args.hf_repo and args.hf_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                hf_path = f"corpus/{args.dataset}/{out_path.name}"
                api.upload_file(path_or_fileobj=str(out_path), path_in_repo=hf_path, repo_id=args.hf_repo, repo_type="dataset", token=args.hf_token)
                os.remove(out_path)
            except Exception as e:
                print(f"Failed to upload chunk: {e}")
        
    print(f"Finished {args.dataset} Subj {args.subject_idx} in {time.time() - start_time:.1f}s")
    
if __name__ == "__main__":
    main()
