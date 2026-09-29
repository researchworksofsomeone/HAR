#!/bin/bash
#SBATCH --job-name=perconai_loso
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1

# MOLAB Job Submission Script
echo "Starting PerconAI Job on MOLAB"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Use absolute paths
export WORKDIR=$SLURM_SUBMIT_DIR
cd $WORKDIR

# Load python module if required by MOLAB
# module load python/3.10

# Thread Capping for Determinism
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

# Optional: route data/checkpoints to local scratch if available
# export PERCONAI_DATA_DIR=/scratch/$USER/$SLURM_JOB_ID/data
# export PERCONAI_CKPT_DIR=/scratch/$USER/$SLURM_JOB_ID/checkpoints

# Activate environment (assuming venv)
source .venv/bin/activate

# Execute the pipeline (Days 1-3)
# Note: In production, the notebook should be converted to a script or run via papermill/nbconvert
jupyter nbconvert --to notebook --execute notebooks/PerconAI_Days_1_3.ipynb --output notebooks/PerconAI_Days_1_3_executed.ipynb

echo "Job Completed"
