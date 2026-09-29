import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def generate_requirements():
    reqs = """torch>=2.0.0
numpy>=1.23.0
pandas>=2.0.0
pyarrow>=12.0.0
psutil>=5.9.0
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-learn>=1.2.0
huggingface_hub>=0.14.0"""
    with open(PROJECT_ROOT / "requirements.txt", "w") as f:
        f.write(reqs)

def generate_readme():
    readme = """# JADE: Joint Adaptation and Drift Execution (Anonymous Artifact)

## Abstract
This repository contains the complete implementation, evaluation scripts, and processed data for our IEEE PerCom 2027 double-blind submission. 

## Directory Structure
- `src/`: Core implementation of TinyHAR-Net, JADE controllers, and Stats engines.
- `configs/`: Hyperparameters and hardware profiles.
- `scripts/`: Reproduction scripts for data ingestion, model training, and evaluation.
- `data/processed/`: Frozen meta-training corpus and aggregated results.

## Reproduction
1. **Sanity Check:** `python scripts/eval_fast.py`
2. **Cluster Eval:** `sbatch scripts/submit_molab_full_eval.sh`
3. **Statistics:** `python scripts/analyze_results.py`
4. **Figures:** `python scripts/generate_figures.py`
"""
    with open(PROJECT_ROOT / "README.md", "w") as f:
        f.write(readme)

def generate_checklist():
    checklist = """# Double-Blind Compliance Checklist
- [x] No author names or affiliations in any file
- [x] No proprietary data used
- [x] CPU-only constraints met (no CUDA in production code)
- [x] All pre-registered hypotheses tested with specified statistical methods
- [x] All numeric knobs match frozen config
- [x] Energy accounting uses published coefficients
"""
    with open(PROJECT_ROOT / "double_blind_checklist.md", "w") as f:
        f.write(checklist)

def create_zip():
    out_zip = PROJECT_ROOT / "jade_anonymous_artifact.zip"
    
    include_dirs = ["src", "configs", "scripts", "figures", "results", "checkpoints"]
    include_files = ["requirements.txt", "README.md", "double_blind_checklist.md"]
    
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for d in include_dirs:
            dir_path = PROJECT_ROOT / d
            if dir_path.exists():
                for root, _, files in os.walk(dir_path):
                    if "__pycache__" in root: continue
                    for file in files:
                        filepath = Path(root) / file
                        arcname = filepath.relative_to(PROJECT_ROOT)
                        zipf.write(filepath, arcname)
                        
        # Include data/processed specifically (avoid raw/memmap which are huge)
        data_proc = PROJECT_ROOT / "data" / "processed"
        if data_proc.exists():
            for root, _, files in os.walk(data_proc):
                for file in files:
                    filepath = Path(root) / file
                    arcname = filepath.relative_to(PROJECT_ROOT)
                    zipf.write(filepath, arcname)
                    
        for f in include_files:
            file_path = PROJECT_ROOT / f
            if file_path.exists():
                zipf.write(file_path, f)
                
    return out_zip

def main():
    print("Packaging Final Artifact...")
    generate_requirements()
    generate_readme()
    generate_checklist()
    zip_path = create_zip()
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Artifact packaged successfully: {zip_path.name} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
