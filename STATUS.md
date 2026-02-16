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
| Parallel self-play engine | **Fixed** (state-ref bug resolved) |
| Training loop (train_alphazero.py) | **Fixed** (value bootstrap + action limit) |
| Config system (Quick/Small/Medium/H100/H200) | Working |
| PACE/ICE deployment scripts | Written, untested on cluster |

## Bugs Found & Fixed (Feb 2026)

### Bug 1: State reference corruption in parallel self-play (CRITICAL)
`parallel_selfplay.py` stored `states.append(state)` during game play, but `apply_action()` mutates state in-place. ALL entries in the `states` list pointed to the same final GameState. This meant EVERY training example from parallel self-play had identical features (the final state) paired with DIFFERENT policies. The network was being trained on garbage data.

**Fix:** Pre-encode features at the time of generation in `_play_single_game()`. Store `{features, player, policy, legal_indices}` dicts instead of raw state references.

### Bug 2: Self-play action limit too low
`max_turns = 500` counted every ACTION (not game turns). Catan needs 600-1500 actions to complete. 100% of games hit the limit → no completed games → no training signal.

**Fix:** Increased `max_turns` to 2000 in both `train_alphazero.py` and `parallel_selfplay.py`.

### Bug 3: Zero value signal from untrained NN
AlphaZero MCTS replaces rollouts with NN value. Untrained NN returns ~0 for all positions, making MCTS search effectively random (0% win rate vs random agents). Regular MCTS with rollouts wins 80%.

**Fix:** Blend NN value with VP-based heuristic: `value = (1-w)*nn_value + w*(VP/10)`. Weight `w` starts at 0.8 and decays linearly to 0 over training iterations. Heuristic is also applied in the batched evaluator by extracting VP from encoded features at known offsets.

### Bug 4: Incomplete games discarded
`generate_self_play_data()` discarded all training examples from incomplete games. Combined with Bug 2, this meant zero training data.

**Fix:** Use VP-proxy values (`VP/10`) for incomplete games instead of discarding them.

## Verification

Integration test (2 games, 10 MCTS sims):
- **100% game completion** (avg 831 turns per game)
- **1663 training examples** generated (previously 0)
- **Features correctly encoded** (differ between examples — state-ref bug confirmed fixed)
- **26.4% non-zero values** in training data
- **Policy loss: 3.26, Value loss: 0.04** — both meaningful

## Ready for Training

The training pipeline is now functional. To run:
```bash
cd AlphaSettler
conda activate alphasettler
python python/train_alphazero.py
```
Default config: QuickTest (5 iters × 20 games = 100 games). Estimated 10-30 min on CPU.

## Next Steps (Priority Order)

1. **Run QuickTest training** — verify loss curves decrease and games complete
2. **Evaluate trained vs random** — verify the network actually beats random after training
3. **Tune training** — adjust heuristic decay rate, learning rate, MCTS sims
4. **Scale up** — SmallTraining (500 games), then deploy to PACE for GPU training

## Config: Heuristic Value Bootstrap

New config parameters in `config.py`:
- `heuristic_value_weight = 0.8` — initial blend weight (0=all NN, 1=all heuristic)
- `heuristic_decay_iterations = None` — linear decay over N iterations (default: all iterations)

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
