# AlphaSettler

**AlphaZero-style deep reinforcement learning for Settlers of Catan**

Fast C++ game engine with batched GPU evaluation, parallel self-play, and neural network training.

## Quick Start

### 1. Build
```bash
# Install dependencies
pip install torch numpy tqdm

# Build C++ engine
python setup.py build_ext --inplace
```

### 2. Test
```bash
# Complete end-to-end test
python test_complete.py
```

### 3. Train
```bash
# Local training
python python/train_alphazero.py

# Edit python/config.py to choose:
# - QuickTestConfig (3 iter, 15 games)
# - SmallTrainingConfig (20 iter, 1K games)
# - MediumTrainingConfig (100 iter, 10K games)
# - H100Config / H200Config (GPU optimized)
```

### 4. Deploy to PACE/ICE
```bash
# Copy files
scp -r AlphaSettler <username>@login-ice.pace.gatech.edu:~/

# SSH and setup
ssh <username>@login-ice.pace.gatech.edu
cd ~/AlphaSettler
chmod +x deploy_pace.sh
./deploy_pace.sh

# Submit training job
sbatch run_training.sh
```

## Features

- **C++ Game Engine**: Efficient board state and move generation
- **AlphaZero MCTS**: Neural network-guided tree search
- **Batched Evaluation**: 2-5x speedup via batch NN inference
- **Parallel Self-Play**: Multi-threaded game generation
- **GPU Training**: PyTorch with automatic H100/H200 detection
- **Progress Monitoring**: Real-time progress bars and metrics

## Training Configs

| Config | Iterations | Games | Time (H100) |
|--------|-----------|--------|-------------|
| QuickTest | 3 | 15 | ~1 min |
| Small | 20 | 1K | ~10 min |
| Medium | 100 | 10K | ~1 hour |
| H100 | 2000 | 2M | ~2-5 days |
| H200 | 3000 | 4.5M | ~3-7 days |

## Output

Training produces:
```
training_runs/YYYYMMDD_HHMMSS/
├── training_log.json       # All metrics
├── run_info.json          # Config info
├── training_curves.png    # Loss plots
└── checkpoints/
    ├── checkpoint_iter_100.pt
    └── final_model.pt
```

## Project Structure

```
AlphaSettler/
├── include/              # C++ headers
│   ├── action.h
│   ├── game_state.h
│   ├── move_gen.h
│   └── mcts/            # MCTS headers
├── src/                 # C++ implementation
│   ├── game_state.cpp
│   ├── move_gen.cpp
│   └── mcts/
├── python/
│   ├── train_alphazero.py      # Main training loop
│   ├── parallel_selfplay.py    # Multi-threaded self-play
│   ├── catan_network.py        # Neural network
│   ├── config.py              # Training configs
│   └── bindings.cpp           # Python/C++ interface
├── test_complete.py     # End-to-end test
├── deploy_pace.sh      # PACE setup script
└── run_training.sh     # SLURM job script
```

## Performance

- **Local (CPU, 8 cores)**: ~2-5 games/sec
- **H100 GPU**: ~15-30 games/sec (32 workers, batch 128)
- **H200 GPU**: ~20-40 games/sec (48 workers, batch 192)
- **Batch efficiency**: 2.0-5.0 avg batch size

## Requirements

- **C++17** compiler (GCC 9+, MSVC 2019+)
- **Python 3.11+**
- **PyTorch** with CUDA 12.1+ (for GPU)
- **pybind11**, **numpy**, **tqdm**


