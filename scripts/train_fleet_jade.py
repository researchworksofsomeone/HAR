import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env_setup import init_environment
from scripts.train_predictor import LogisticPredictor, ece_score

def main():
    print("==================================================")
    print("  DAY 15: FLEET-JADE Factory Calibration")
    print("==================================================")
    
    cfg_path = str(PROJECT_ROOT / "configs" / "frozen.yaml")
    cfg, device, ckpt_mgr = init_environment(cfg_path)
    
    corpus_path = PROJECT_ROOT / "data" / "processed" / "meta_train_corpus_frozen.parquet"
    if not corpus_path.exists():
        print(f"Corpus not found at {corpus_path}. Creating dummy data for dry run...")
        df = pd.DataFrame({
            "dataset": ["UCI-HAR", "PAMAP2", "HHAR"] * 100,
            "phi_t": [np.random.randn(28).tolist() for _ in range(300)],
            "action_id": np.random.randint(0, 4, 300),
            "bin_label": np.random.choice(["harm", "neutral", "help"], 300)
        })
    else:
        df = pd.read_parquet(corpus_path)
        
    print(f"Loaded Meta-Corpus: {len(df)} samples.")
    
    # For FLEET-JADE, we train on the UNION of all datasets (controller-training subjects).
    # Since meta_train_corpus_frozen only contains controller-training subjects, we use all of it.
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    label_map = {"harm": 0, "neutral": 1, "help": 2}
    
    train_df["bin_label_mapped"] = train_df["bin_label"].astype(str).str.lower().str.strip().map(label_map)
    val_df["bin_label_mapped"] = val_df["bin_label"].astype(str).str.lower().str.strip().map(label_map)
    
    train_df = train_df.dropna(subset=["bin_label_mapped"])
    val_df = val_df.dropna(subset=["bin_label_mapped"])
    
    if len(train_df) == 0:
        print("❌ CRITICAL: No valid training data left after label mapping!")
        sys.exit(1)
        
    X_train = torch.tensor(np.vstack(train_df["phi_t"].values), dtype=torch.float32)
    y_train = torch.tensor(train_df["bin_label_mapped"].values, dtype=torch.long)
    
    X_val = torch.tensor(np.vstack(val_df["phi_t"].values), dtype=torch.float32)
    y_val = torch.tensor(val_df["bin_label_mapped"].values, dtype=torch.long)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # FLEET-JADE uses STRICT Multinomial Logistic (0 hidden layers) for minimal overhead
    in_features = X_train.shape[1]
    model = LogisticPredictor(in_features).to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 50
    best_loss = float('inf')
    
    print("Training FLEET-JADE Predictor...")
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_loss = criterion(val_logits, y_val.to(device)).item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict().copy()
                
    model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val.to(device))
        probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        acc = (preds == y_val.numpy()).mean()
        ece = ece_score(probs, y_val.numpy())
        
    print(f"[FLEET-JADE] Validation Accuracy: {acc*100:.2f}% | ECE: {ece:.4f}")
    
    out_dir = PROJECT_ROOT / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "predictor_fleet_jade.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved Universal Weights to {out_path}")
    
    # Regret Reduction Check (Proxy for PH4)
    if acc > 0.33:
        print("✓ FLEET-JADE Regret Reduction Proxy Check Passed (Acc > Random).")
    else:
        print("⚠ Warning: FLEET-JADE predictor may struggle with high regret.")

if __name__ == "__main__":
    main()
