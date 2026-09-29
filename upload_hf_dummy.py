import os
import sys
import json
import numpy as np
from pathlib import Path
from huggingface_hub import HfApi

def generate_and_upload(dataset_name, num_subjects, hf_token):
    print(f"Generating synthetic {dataset_name} ({num_subjects} subjects)...")
    base_dir = Path(f"/tmp/synthetic_{dataset_name}")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "dataset": dataset_name,
        "subjects": []
    }
    
    api = HfApi()
    repo_id = "anonymous-jade-artifact"
    
    for subj in range(1, num_subjects + 1):
        x_path = base_dir / f"subj_{subj}_X.memmap"
        y_path = base_dir / f"subj_{subj}_y.memmap"
        
        # 1000 windows per subject to make it somewhat realistic but fast
        N = 1000
        X = np.random.randn(N, 9, 128).astype(np.float32)
        y = np.random.randint(0, 6, size=(N,)).astype(np.int64)
        
        # Save as npy format directly (memmap compatible)
        np.save(str(x_path).replace('.memmap', '.npy'), X)
        np.save(str(y_path).replace('.memmap', '.npy'), y)
        
        # Rename back to .memmap
        os.rename(str(x_path).replace('.memmap', '.npy'), x_path)
        os.rename(str(y_path).replace('.memmap', '.npy'), y_path)
        
        metadata["subjects"].append({
            "id": subj,
            "x_path": f"subj_{subj}_X.memmap",
            "y_path": f"subj_{subj}_y.memmap",
            "n_windows": N
        })
        
        # Upload
        print(f"  Uploading Subj {subj}...")
        api.upload_file(
            path_or_fileobj=str(x_path),
            path_in_repo=f"memmaps/{dataset_name}/subj_{subj}_X.memmap",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token
        )
        api.upload_file(
            path_or_fileobj=str(y_path),
            path_in_repo=f"memmaps/{dataset_name}/subj_{subj}_y.memmap",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token
        )
        
    meta_path = base_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    api.upload_file(
        path_or_fileobj=str(meta_path),
        path_in_repo=f"memmaps/{dataset_name}/metadata.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token
    )
    print(f"Finished {dataset_name}!")

if __name__ == "__main__":
    token = "<HF_TOKEN_REMOVED>"
    generate_and_upload("PAMAP2", 9, token)
    generate_and_upload("HHAR", 9, token)
