#!/bin/bash
# Interactive training script for VS Code remote session
# Run this directly in your terminal when you have GPU access

set -e

echo "=================================================="
echo "AlphaSettler Interactive Training"
echo "=================================================="
echo ""

# Check if we're in conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  Not in conda environment. Activating alphasettler..."
    source ~/.bashrc
    conda activate alphasettler
fi

# Verify GPU access
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "✓ GPU detected: $GPU_NAME"
    echo ""
else
    echo "⚠️  No GPU detected. Training will use CPU (very slow)."
    echo ""
fi

# Auto-detect configuration based on GPU
if [[ -z "${ALPHASETTLER_CONFIG}" ]]; then
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q "H200"; then
        export ALPHASETTLER_CONFIG="H200"
        echo "🚀 Auto-detected: H200 GPU → Using H200Config"
    elif nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q "H100"; then
        export ALPHASETTLER_CONFIG="H100"
        echo "🚀 Auto-detected: H100 GPU → Using H100Config"
    elif nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q "A100"; then
        export ALPHASETTLER_CONFIG="PACE"
        echo "🚀 Auto-detected: A100 GPU → Using PACEGPUConfig"
    elif nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q "L40S"; then
        export ALPHASETTLER_CONFIG="PACE"
        echo "🚀 Auto-detected: L40S GPU → Using PACEGPUConfig"
    else
        export ALPHASETTLER_CONFIG="QuickTest"
        echo "⚠️  No recognized GPU → Using QuickTestConfig (for testing)"
    fi
else
    echo "📝 Using config from environment: $ALPHASETTLER_CONFIG"
fi

echo ""
echo "Configuration: $ALPHASETTLER_CONFIG"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "=================================================="
echo "Starting training..."
echo "=================================================="
echo ""

# Run training with output to both console and file
# python python/train_alphazero.py 2>&1 | tee training_interactive.log

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
echo "Log saved to: training_interactive.log"
echo "Models saved to: training_runs/"
