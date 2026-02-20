# AlphaSettler

AlphaZero-style reinforcement learning for Catan, built on a custom C++17 engine with Python training and evaluation tooling.

This repo is a full-stack RL project: game rules engine, move generation, search, neural network inference, parallel self-play, training, and evaluation.

## What I Built

- A full Catan rules engine in C++ (`game_state`, transitions, legal move generation, scoring, dev cards, robber, trading, setup flow)
- A compact action system (`Action` is 8 bytes) and copy-friendly game state designed for search speed
- Two search stacks:
  - Pure MCTS with rollouts
  - AlphaZero MCTS (PUCT + policy/value guidance + Dirichlet root noise + virtual loss)
- A pybind11 bridge (`python/bindings.cpp`) so the C++ engine runs directly from Python
- A batched NN evaluator and multithreaded self-play pipeline for higher throughput
- A residual policy/value network in PyTorch (`python/catan_network.py`)
- End-to-end training + checkpointing + evaluation tooling

## Engineering Highlights

- **Rule-complete action generation:** Setup placements, roads/settlements/cities, dev cards, robber actions, discards, bank and port trades
- **Perspective-aware state encoding:** C++ encoder exposes hidden info for self and public info for opponents
- **Parallelism where it matters:** Multiple self-play workers share one batched evaluator
- **Search quality controls:** PUCT exploration, root noise, and visit-count policies for training targets
- **Practical training pipeline:** Replay buffer, policy+value losses, checkpoint cadence, metrics logging, training curves

## Recent Improvements


- Fixed a critical parallel self-play data bug (state references were mutating in-place)
- Raised action cap for realistic game completion (2000 actions)
- Added value bootstrapping (NN value blended with VP heuristic early in training)
- Kept incomplete-game examples by assigning VP-proxy values instead of discarding data

## Quick Start

### 1. Install + build

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

### 2. Run an evaluation matchup

```bash
python python/eval_agents.py --games 20 --iterations 100 --mode mcts_vs_random
```

### 3. (Optional) Run training

```bash
python python/train_alphazero.py
```

## C++ Test Binaries

```bash
cmake -S . -B build
cmake --build build
```

Built test executables include:

- `test_game_state`
- `test_move_gen`
- `test_state_transition`
- `test_mcts`
- `test_mcts_agent`
- `test_visualize_grid`

## Project Layout

```text
AlphaSettler/
|-- include/              # C++ headers (engine, rules, search, encoder)
|-- src/                  # C++ implementation
|   `-- mcts/             # MCTS + AlphaZero MCTS
|-- python/
|   |-- bindings.cpp      # pybind11 bridge to C++
|   |-- catan_network.py  # PyTorch policy/value network
|   |-- parallel_selfplay.py
|   |-- train_alphazero.py
|   |-- eval_agents.py
|   `-- eval_alphazero.py
|-- tests/                # C++ test programs
|-- _eval_checkpoints.py
|-- CMakeLists.txt
`-- setup.py
```

## Tech Stack

- C++17 (engine + search)
- pybind11 (bindings)
- Python + PyTorch (training/inference)
- NumPy + tqdm + matplotlib (data and instrumentation)
