#!/bin/bash
#SBATCH --job-name=alphasettler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=training_%j.out

echo "=== AlphaSettler Training Job Starting ==="

# Go to the directory you submitted from (AlphaSettler/)
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

# Load modules
module load anaconda3/2023.03
module load cuda/12.1

# Activate env created by deploy_pace.sh
source activate alphasettler

echo "Using Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Choose training config + base run dir (optional but recommended)
export ALPHASETTLER_CONFIG=H100Config         # or QuickTestConfig / PACEGPUConfig / etc.
export ALPHASETTLER_RUN_DIR=/scratch/ice/$USER/alphasettler_runs
mkdir -p "$ALPHASETTLER_RUN_DIR"

echo "Run dir base: $ALPHASETTLER_RUN_DIR"
echo "Config: $ALPHASETTLER_CONFIG"

echo "=== Detecting GPU ==="
nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "Detected GPU: $GPU_NAME"

echo "=== Starting Training ==="
# NOTE: path is relative to AlphaSettler/, where we cd'd above
python python/train_alphazero.py

echo "=== Training Completed ==="