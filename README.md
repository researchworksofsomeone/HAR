# JADE: Joint Adaptation and Drift Execution (Anonymous Artifact)

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
