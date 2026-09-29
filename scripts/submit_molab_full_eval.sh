#!/bin/bash
#SBATCH --job-name=jade_full_eval
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-7560%50
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

module load python/3.10
source ../.venv/bin/activate
export OMP_NUM_THREADS=2

# Use node-local scratch for blazing fast I/O
export TMPDIR="/scratch/$USER/jade_eval_tmp"
mkdir -p $TMPDIR

MANIFEST="data/results/task_manifest.csv"
FINAL_DIR="$HOME/PerconAI/data/results/raw_eval_chunks"
mkdir -p $FINAL_DIR

echo "Starting Eval Task $SLURM_ARRAY_TASK_ID on $HOSTNAME"

python scripts/eval_worker.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --manifest_path $MANIFEST \
    --output_dir $TMPDIR

# Rsync individual result to persistent storage
cp $TMPDIR/result_task${SLURM_ARRAY_TASK_ID}.parquet $FINAL_DIR/

echo "Eval Task $SLURM_ARRAY_TASK_ID finished."
