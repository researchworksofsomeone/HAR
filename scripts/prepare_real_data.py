import argparse
import os
import sys
import gc
import json
import zipfile
import urllib.request
import hashlib
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def download_with_retry(url, dest, checksum=None, retries=3):
    for attempt in range(retries):
        try:
            print(f"Downloading {url} (Attempt {attempt+1}/{retries})...")
            urllib.request.urlretrieve(url, dest)
            if checksum:
                hash_md5 = hashlib.md5()
                with open(dest, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                if hash_md5.hexdigest() != checksum:
                    raise ValueError("Checksum mismatch!")
            print("Download successful.")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
    return False

def parse_uci_har(raw_dir: Path, out_dir: Path, hf_repo=None, hf_token=None):
    """Parses REAL UCI-HAR from unzipped folder into isolated float32 memmaps."""
    print("Parsing REAL UCI-HAR...")
    
    extract_dir = raw_dir / "UCI HAR Dataset"
    if not extract_dir.exists():
        # Try unzipping
        zip_path = raw_dir / "uci_har.zip"
        if zip_path.exists():
            print("Unzipping UCI-HAR...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
        else:
            raise FileNotFoundError("UCI-HAR dataset not found. Download it first.")
            
    channels_files = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    
    def load_group(group_name):
        group_dir = extract_dir / group_name
        signals_dir = group_dir / "Inertial Signals"
        
        # Load labels and subjects
        y_path = group_dir / f"y_{group_name}.txt"
        subj_path = group_dir / f"subject_{group_name}.txt"
        
        y = np.loadtxt(y_path, dtype=np.int64) - 1 # 1-indexed to 0-indexed
        subjects = np.loadtxt(subj_path, dtype=np.int64)
        
        # Load channels
        X_channels = []
        for ch in channels_files:
            ch_path = signals_dir / f"{ch}_{group_name}.txt"
            X_ch = np.loadtxt(ch_path, dtype=np.float32)
            X_channels.append(X_ch)
            
        X = np.stack(X_channels, axis=1) # (N, 9, 128)
        return X, y, subjects

    print("Loading train set...")
    X_train, y_train, subj_train = load_group("train")
    print("Loading test set...")
    X_test, y_test, subj_test = load_group("test")
    
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    subj_all = np.concatenate([subj_train, subj_test], axis=0)
    
    unique_subjects = np.unique(subj_all)
    metadata = {"dataset": "UCI-HAR", "subjects": []}
    
    for subj in unique_subjects:
        print(f"Processing UCI-HAR Subject {subj}...")
        
        mask = (subj_all == subj)
        X_subj = X_all[mask]
        y_subj = y_all[mask]
        n_windows = len(X_subj)
        
        # Z-score normalization per channel
        mean = X_subj.mean(axis=(0, 2), keepdims=True)
        std = X_subj.std(axis=(0, 2), keepdims=True) + 1e-8
        X_subj = (X_subj - mean) / std
        
        # Save to memmap
        x_filename = f"subj_{subj}_X.memmap"
        y_filename = f"subj_{subj}_y.memmap"
        x_path = out_dir / x_filename
        y_path = out_dir / y_filename
        
        X_mmap = np.lib.format.open_memmap(x_path, mode='w+', dtype=np.float32, shape=X_subj.shape)
        X_mmap[:] = X_subj[:]
        X_mmap.flush()
        del X_mmap
        
        y_mmap = np.lib.format.open_memmap(y_path, mode='w+', dtype=np.int64, shape=y_subj.shape)
        y_mmap[:] = y_subj[:]
        y_mmap.flush()
        del y_mmap
        
        metadata["subjects"].append({
            "id": int(subj),
            "x_path": x_filename,
            "y_path": y_filename,
            "windows": n_windows
        })
        
        # HuggingFace Upload Logic to save space
        if hf_repo and hf_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                hf_x_path = f"memmaps/UCI-HAR/{x_filename}"
                hf_y_path = f"memmaps/UCI-HAR/{y_filename}"
                
                api.upload_file(path_or_fileobj=str(x_path), path_in_repo=hf_x_path, repo_id=hf_repo, repo_type="dataset", token=hf_token)
                api.upload_file(path_or_fileobj=str(y_path), path_in_repo=hf_y_path, repo_id=hf_repo, repo_type="dataset", token=hf_token)
                
                os.remove(x_path)
                os.remove(y_path)
                print(f"Uploaded {x_filename} to HF and cleared local.")
            except Exception as e:
                print(f"HF Upload Failed for Subj {subj}: {e}")
                
        gc.collect() # STRICT MEMORY ISOLATION
        
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    if hf_repo and hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(path_or_fileobj=str(meta_path), path_in_repo="memmaps/UCI-HAR/metadata.json", repo_id=hf_repo, repo_type="dataset", token=hf_token)
            print("Uploaded metadata.json to HF.")
            os.remove(meta_path)
        except Exception as e:
            pass
            
        import shutil
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        zip_path = raw_dir / "uci_har.zip"
        if zip_path.exists():
            os.remove(zip_path)
        print("Cleaned up raw dataset files.")

def main():
    parser = argparse.ArgumentParser(description="Headless Real Data Preparation")
    parser.add_argument("--dataset", type=str, choices=["UCI-HAR", "PAMAP2", "HHAR"], required=True)
    parser.add_argument("--output_dir", type=str, default="/tmp/jade_data")
    parser.add_argument("--max_subjects", type=int, default=None, help="For dry runs")
    parser.add_argument("--hf_repo", type=str, default=None, help="HuggingFace dataset repo (e.g., anonymous-jade-artifact)")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace write token")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = Path("/tmp/jade_raw") / args.dataset
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == "UCI-HAR":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        zip_path = raw_dir / "uci_har.zip"
        if not zip_path.exists():
            download_with_retry(url, zip_path)
        parse_uci_har(raw_dir, out_dir, hf_repo=args.hf_repo, hf_token=args.hf_token)
    else:
        print(f"Dataset {args.dataset} parsing logic placeholder.")

if __name__ == "__main__":
    main()
