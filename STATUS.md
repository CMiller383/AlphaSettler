# AlphaSettler - Project Status

**Last verified: Feb 2026**

## What Works

| Component | Status |
|-----------|--------|
| C++ game engine (state, moves, transitions) | Working |
| Python bindings (pybind11) | Working |
| MCTS search (AlphaZero-style) | Working |
| Neural network (CatanNetwork, ResNet) | Working |
| State encoder (C++ and Python) | Working |
| Batched NN evaluator (C++) | Working (multithreaded only) |
| Parallel self-play engine | Working |
| Training loop (train_alphazero.py) | Working |
| Config system (Quick/Small/Medium/H100/H200) | Working |
| PACE/ICE deployment scripts | Written, untested on cluster |

## What Doesn't Work Well

**Random games don't complete** - even a random agent hits the 500-turn limit without reaching 10 VP. This suggests possible game logic issues (resource generation, trading, or win-condition bugs) OR that random play legitimately can't finish Catan.

**Training produces no meaningful learning:**
- 5 training iterations x 20 games completed (the last run, Nov 2022)
- Policy loss dropped 4.1 -> 0.75, value loss dropped 0.55 -> 0.0002
- But games never complete -- agents get stuck ending turn repeatedly
- The trained model is no better than random

## Root Cause (Previously Diagnosed)

1. **Untrained network biases toward EndTurn** - uniform policy + EndTurn always available = MCTS gets stuck in EndTurn loops
2. **Training on incomplete games** - 95%+ of training data is from stuck 500-turn games, so the network learns "EndTurn is good"
3. **AlphaZero MCTS bug** - partially-expanded nodes get re-evaluated instead of explored deeper, causing shallow search trees (identified in `MCTS_BUG_ANALYSIS.md`, may or may not be fixed)

## Proposed Fixes (Never Implemented)

1. **Heuristic policy bias** - boost building actions, suppress EndTurn in early training
2. **Training data filtering** - only train on completed games or games showing VP progress
3. **Epsilon-greedy exploration** - mix random moves early, gradually trust network
4. **Reward shaping** - partial credit for VP progress instead of binary win/lose
5. **Performance optimizations** - pre-stacked arrays in C++, virtual loss for parallel MCTS (see `OPTIMIZATION_PLAN.md`)

## Next Steps (Priority Order)

1. **Verify game completability** - can ANY agent (even heuristic) reliably reach 10 VP? If not, there may be a game logic bug
2. **Fix the MCTS re-evaluation bug** if not already fixed (check `alphazero_mcts.cpp` lines ~241-250)
3. **Implement heuristic policy bias** for early training iterations
4. **Add reward shaping** (VP-based partial rewards)
5. **Run a real training session** on GPU with Small or Medium config
6. **Deploy to PACE** for full-scale training once local tests show learning progress

## Project Structure (Clean)

```
AlphaSettler/
├── include/                    # C++ headers
│   ├── action.h, game_state.h, move_gen.h, board_grid.h
│   ├── player_state.h, resources.h, dev_cards.h
│   ├── state_encoder.h, state_transition.h, batched_evaluator.h
│   └── mcts/                   # alphazero_mcts.h, mcts_agent.h, mcts_node.h, mcts_search.h
├── src/                        # C++ implementation
│   ├── game_state.cpp, move_gen.cpp, board_grid.cpp
│   ├── state_encoder.cpp, state_transition.cpp, batched_evaluator.cpp
│   └── mcts/                   # alphazero_mcts.cpp, mcts_agent.cpp, mcts_search.cpp
├── python/
│   ├── bindings.cpp            # pybind11 C++/Python bridge
│   ├── catan_network.py        # ResNet neural network
│   ├── config.py               # Training configs (Quick/Small/Medium/H100/H200)
│   ├── state_encoder.py        # Python state encoder
│   ├── train_alphazero.py      # Main training loop
│   ├── parallel_selfplay.py    # Multi-threaded self-play
│   ├── eval_agents.py          # Agent evaluation
│   ├── eval_alphazero.py       # AlphaZero evaluation
│   └── agents/                 # Agent wrappers (alphazero_agent.py, random_agent.py)
├── tests/                      # C++ unit tests
├── setup.py                    # Build script
├── CMakeLists.txt              # CMake config
├── deploy_pace.sh              # PACE cluster setup
├── run_training.sh             # SLURM job script
└── requirements.txt            # Python deps
```
