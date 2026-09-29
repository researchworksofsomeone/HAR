#!/bin/bash
#SBATCH --job-name=jade_corpus_gen
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --array=0-149%20
#SBATCH --output=logs/corpus_%A_%a.out
#SBATCH --error=logs/corpus_%A_%a.err

# Load environment
module load python/3.10
source ../.venv/bin/activate
export OMP_NUM_THREADS=2

# If compute cluster queue times exceed 48 hours, this specific R6/HHAR block can be executed on a T4 GPU 
# by changing --gres=gpu:1 and ensuring device='cuda' in the script, though CPU logic remains identical.

# Directories
export SCRATCH_DIR="/scratch/$USER/jade_data"
export FINAL_DIR="$HOME/PerconAI/data/corpus"

mkdir -p $SCRATCH_DIR
mkdir -p $FINAL_DIR

# ---------------------------------------------------------------------------
# Map SLURM_ARRAY_TASK_ID to Job Config
# ---------------------------------------------------------------------------
# We have ~150 jobs (3 datasets x ~6 regimes x ~8 subjects).
# For production compute cluster run, we map the ID explicitly via a lookup table or python helper.
# Placeholder bash logic:
DATASET="UCI-HAR"
REGIME="R1"
SUBJECT_IDX=$((SLURM_ARRAY_TASK_ID + 1))
SEVERITY="mid"

# For R6 specifically, we need to pass the severity argument
if [ "$REGIME" = "R6" ]; then
    SEVERITY_ARG="--severity $SEVERITY"
else
    SEVERITY_ARG=""
fi

echo "Starting Task $SLURM_ARRAY_TASK_ID | Subj $SUBJECT_IDX | Node $HOSTNAME"

# 1. Preprocess Data to fast local scratch (if not already done)
python scripts/prepare_real_data.py --dataset $DATASET --output_dir $SCRATCH_DIR

# 2. Run Corpus Generation
python scripts/generate_corpus.py \
    --dataset $DATASET \
    --regime $REGIME \
    --subject_idx $SUBJECT_IDX \
    --output_dir $SCRATCH_DIR \
    $SEVERITY_ARG

# 3. Rsync back to persistent storage
echo "Copying Parquet chunks to home directory..."
rsync -av $SCRATCH_DIR/corpus/$DATASET/ $FINAL_DIR/$DATASET/

echo "Job completed successfully."
