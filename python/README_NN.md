# Neural Network Implementation for Catan AlphaZero

## Overview

This directory contains the neural network and training infrastructure for the Catan AlphaZero agent.

## Components

### 1. State Encoder (`state_encoder.py`)
Converts game state into ~900-dimensional feature vector:
- **Board state**: Tile resources, dice numbers, robber position
- **Pieces**: Settlements, cities, roads (per player)
- **Player state**: Resources, dev cards, pieces remaining, VP
- **Global info**: Current player, game phase, special achievements

### 2. Neural Network (`catan_network.py`)
ResNet-style architecture:
- Input: State features (~900 dims)
- Trunk: 4 residual blocks with 512 hidden units
- Policy head: Outputs action logits (max 300 actions, masked)
- Value head: Outputs win probability [-1, 1]

### 3. Training Loop (`train_alphazero.py`)
Full AlphaZero training cycle:
1. **Self-play**: Generate games with MCTS + current network
2. **Collect**: Store (state, policy, value) training examples
3. **Train**: Update network on replay buffer
4. **Evaluate**: Test against baselines
5. **Checkpoint**: Save best models

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Build C++ Engine
```bash
cd build
cmake --build . --config Release
cd ..
pip install -e .
```

### Run Integration Tests
```bash
python python/test_nn_integration.py
```

Tests:
- State encoder shape and validity
- Network forward pass
- Network prediction interface
- AlphaZero MCTS with NN
- Complete game playthrough

### Start Training
```bash
python python/train_alphazero.py
```

Default configuration:
- 100 iterations
- 100 games per iteration (self-play)
- 200 MCTS simulations per move
- 5 training epochs per iteration
- Batch size 256
- Replay buffer 100k examples

Checkpoints saved to `checkpoints/` every 10 iterations.

## Configuration

Edit `TrainingConfig` in `train_alphazero.py`:

```python
config = TrainingConfig()
config.hidden_size = 512           # Network hidden size
config.num_residual_blocks = 4     # ResNet depth
config.learning_rate = 0.001       # Adam LR
config.mcts_simulations = 200      # MCTS sims per move
config.games_per_iteration = 100   # Self-play games
config.batch_size = 256            # Training batch size
```

## Performance Expectations

**Pure MCTS baseline**: ~1.7s/game @ 100 iterations
**With NN (not optimized)**: ~2-5s/game @ 200 simulations
**Target (with batching)**: ~0.5-1.0s/game @ 400 simulations

Training at 100 games/iteration:
- ~5-10 minutes per iteration (depending on hardware)
- 100 iterations = 10,000 games ≈ 8-16 hours

## Next Steps

1. **Run initial training**: 10k games to verify learning
2. **Evaluate vs baselines**: Compare to pure MCTS and random
3. **Optimize performance**: 
   - Batched NN inference
   - Multi-threaded MCTS
   - C++ self-play loop
4. **Scale up**: Train on 100k-1M games

## Files

- `state_encoder.py`: Game state → feature vector
- `catan_network.py`: PyTorch NN (policy + value)
- `train_alphazero.py`: Training loop
- `test_nn_integration.py`: Integration tests
- `eval_agents.py`: Agent evaluation framework
- `agents/`: Agent implementations

## Known Limitations

1. **Action space**: Currently simplified (300 max actions)
   - Full action space is larger and variable
   - Masking handles illegal actions
   - May need factored action space for efficiency

2. **Value target**: Using binary win/loss
   - Could use VP-based partial credit
   - Important for 4-player credit assignment

3. **No batching**: Currently processes states one at a time
   - Major bottleneck for training speed
   - Need to accumulate MCTS leaf nodes

4. **Single-threaded**: MCTS runs on one thread
   - Virtual loss needed for parallelization
   - 4-8x speedup possible

## Troubleshooting

**Import errors**: Ensure C++ extension is built and installed
```bash
pip install -e .
```

**CUDA out of memory**: Reduce batch size or hidden size
```python
config.batch_size = 128
config.hidden_size = 256
```

**Slow training**: Use GPU if available
```python
config.device = torch.device('cuda')
```

**Game not finishing**: Increase max turns in self-play
```python
max_turns = 1000  # In train_alphazero.py
```
