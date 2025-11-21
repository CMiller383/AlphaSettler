#!/bin/bash
#SBATCH --job-name=alphasettler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=training_%j.out

# Load environment
module load anaconda3/2023.03
module load cuda/12.1
source activate alphasettler

# Auto-detect GPU and set config
echo "Detecting GPU..."
nvidia-smi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "Detected GPU: $GPU_NAME"
