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

class LogisticPredictor(nn.Module):
    def __init__(self, in_features, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.linear(x)

class MLPPredictor(nn.Module):
    def __init__(self, in_features, hidden_dim=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def ece_score(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower.item()) * (confidences <= bin_upper.item())
        prop_in_bin = in_bin.astype(float).mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["indomain", "loro", "lodo"], required=True)
    parser.add_argument("--model_type", type=str, choices=["logistic", "mlp"], required=True)
    parser.add_argument("--dataset", type=str, default="UCI-HAR", help="Target dataset for LODO/LORO evaluation")
    parser.add_argument("--regime", type=str, default="R1", help="Target regime for LORO evaluation")
    args = parser.parse_args()
    
    cfg_path = str(PROJECT_ROOT / "configs" / "frozen.yaml")
    cfg, device, ckpt_mgr = init_environment(cfg_path)
    
    corpus_path = PROJECT_ROOT / "data" / "processed" / "meta_train_corpus_frozen.parquet"
    if not corpus_path.exists():
        print(f"Corpus not found at {corpus_path}. Creating dummy data for dry run...")
        # Create a dummy tiny dataframe for structural validation
        df = pd.DataFrame({
            "dataset": ["UCI-HAR"] * 100 + ["PAMAP2"] * 100,
            "regime": ["R1"] * 50 + ["R2"] * 50 + ["R1"] * 100,
            "phi_t": [np.random.randn(28).tolist() for _ in range(200)],
            "action_id": np.random.randint(0, 4, 200),
            "bin_label": np.random.choice(["harm", "neutral", "help"], 200)
        })
    else:
        df = pd.read_parquet(corpus_path)
        
    # Filter based on split
    if args.split == "indomain":
        train_df = df[(df["dataset"] == args.dataset) & (df["regime"] == args.regime)]
        val_df = train_df.sample(frac=0.2, random_state=42)
        train_df = train_df.drop(val_df.index)
    elif args.split == "loro":
        train_df = df[(df["dataset"] == args.dataset) & (df["regime"] != args.regime)]
        val_df = train_df.sample(frac=0.2, random_state=42)
        train_df = train_df.drop(val_df.index)
    elif args.split == "lodo":
        train_df = df[df["dataset"] != args.dataset]
        val_df = train_df.sample(frac=0.2, random_state=42)
        train_df = train_df.drop(val_df.index)

    # Encode labels securely to avoid CUDA out-of-bounds assert on legacy data
    label_map = {"harm": 0, "neutral": 1, "help": 2}
    
    train_df["bin_label_mapped"] = train_df["bin_label"].astype(str).str.lower().str.strip().map(label_map)
    val_df["bin_label_mapped"] = val_df["bin_label"].astype(str).str.lower().str.strip().map(label_map)
    
    # Drop any garbage rows (e.g. from mixed legacy schemas)
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
    
    in_features = X_train.shape[1] # 28
    if args.model_type == "logistic":
        model = LogisticPredictor(in_features).to(device)
    else:
        model = MLPPredictor(in_features).to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 50
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_loss = criterion(val_logits, y_val.to(device)).item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict().copy()
                
    model.load_state_dict(best_state)
    
    # Final metrics
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val.to(device))
        probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        acc = (preds == y_val.numpy()).mean()
        ece = ece_score(probs, y_val.numpy())
        
    print(f"[{args.split} | {args.model_type}] Val Acc: {acc:.4f} | ECE: {ece:.4f}")
    
    out_dir = PROJECT_ROOT / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"predictor_{args.split}_{args.model_type}.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved weights to {out_path}")

if __name__ == "__main__":
    main()
