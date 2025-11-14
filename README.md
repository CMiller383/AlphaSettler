# AlphaSettler - Catan RL Engine

Fast C++ Catan game engine with Python bindings for reinforcement learning.

## Building

### C++ Only (with tests)
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Python Bindings
```bash
pip install pybind11 numpy
pip install -e .
```

## Usage

### Python
```python
from catan_engine import GameState, MCTSAgent, MCTSConfig

# Create game
game = GameState.create_new_game(num_players=4, seed=42)

# Create MCTS agent
config = MCTSConfig()
config.num_iterations = 200
agent = MCTSAgent(config)

# Play game
winner = agent.play_game(game)
print(f"Winner: Player {winner}")
```

### Evaluation
```bash
# Run self-play evaluation
python python/evaluate.py --mode selfplay --games 50 --iterations 200

# Compare configurations
python python/evaluate.py --mode compare --games 20

# Test single game
python python/evaluate.py --mode single --iterations 500
```

## Structure

```
include/           C++ headers
  mcts/           Pure MCTS implementation
src/              C++ source
  mcts/           MCTS implementation
python/           Python bindings and scripts
  bindings.cpp    pybind11 bindings
  evaluate.py     Evaluation scripts
tests/            C++ test executables
```

## Features

- Fast C++ game engine with minimal allocations
- Pure MCTS with random rollouts
- Python bindings via pybind11
- Self-play and evaluation tools
- Ready for neural network integration

## Next Steps

1. Add AlphaZero-style MCTS (with NN priors/values)
2. Implement state encoding for neural networks
3. Build training loop with PyTorch/TensorFlow
