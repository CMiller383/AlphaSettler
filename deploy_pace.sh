#!/bin/bash
# Minimal PACE deployment script

echo "Setting up AlphaSettler on PACE..."

# Load modules
module load anaconda3/2023.03
module load cuda/12.1

# Create and activate environment
conda create -n alphasettler python=3.11 -y
source activate alphasettler

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm

# Build C++ extension
python setup.py build_ext --inplace

# Test installation
echo ""
echo "Testing installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import catan_engine; print('C++ engine: OK')"

echo ""
echo "Setup complete! Now submit a job:"
echo "  sbatch run_training.sh"
