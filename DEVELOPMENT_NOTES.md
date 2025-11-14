# AlphaSettler Development Notes

## Potential Improvements & Optimizations

### Current Known Bottlenecks

#### Performance
1. **Python/C++ Boundary Crossing**
   - Currently crossing ~100-200 times per game (once per action)
   - Move `play_game` to C++ for pure MCTS self-play
   - Keep Python version for NN-guided play and mixed agents

2. **Single-Threaded MCTS**
   - MCTS simulations are embarrassingly parallel
   - Virtual loss technique for multi-threaded tree expansion
   - Could get 4-8x speedup with thread pool

3. **No Batched NN Inference**
   - Currently evaluating one state at a time
   - Batch evaluation of 32-256 states could be 10-100x faster
   - Need to accumulate states from multiple MCTS simulations

#### Game Engine
4. **Longest Road Calculation** ✅
   - ~~Currently TODO/placeholder in `game_state.cpp`~~
   - Implemented with DFS traversal through road network
   - Handles opponent settlements blocking paths

5. **Largest Army Tracking** ✅
   - Implementation verified correct
   - Properly handles tie-breaking (first to reach keeps it)
   - Must exceed current max to steal

6. **Resource Bank Limits** ✅
   - ~~No limits on resource bank currently~~
   - Implemented: 19 of each resource type
   - Enforced on all resource distribution and trading

7. **Development Card Actions** ✅
   - ~~Not generated in move_gen.cpp~~
   - All dev cards now playable: Knight, Road Building, Year of Plenty, Monopoly
   - VP dev cards properly counted toward victory

### Training & Learning

#### Reward Function / Value Target
7. **Binary Win/Loss Only**
   - Current: 1.0 for win, 0.0 for loss
   - Consider: VP-based partial credit (e.g., `vp/10` for incomplete games)
   - Could help with credit assignment in 4-player games

8. **Multi-Player Credit Assignment**
   - 4-player is harder than 2-player zero-sum
   - One player's loss isn't necessarily another's gain
   - May need opponent modeling or self-play curriculum

9. **Exploration vs Exploitation**
   - Current Dirichlet noise: α=0.3, weight=0.25
   - May need tuning for Catan's action space size (~50-200 per state)
   - Consider temperature-based action sampling for diversity

#### State Representation
10. **State Encoding for NN**
    - Need to design: board features, player resources, dev cards, etc.
    - Consider: spatial convolutions for board topology
    - Or: graph neural network for hex grid structure
    - Current: no encoding implemented yet

11. **Action Space Representation**
    - Variable action space (different moves available each turn)
    - Need masking for illegal actions in policy output
    - Could use action templates or factored action space

12. **Hidden Information**
    - Dev cards in hand are hidden from opponents
    - Resources known but not card types unless traded/played
    - May need separate networks for private vs public info

### MCTS Improvements

13. **Pure MCTS Rollout Policy**
    - Current: Biased random (70% building, 30% end turn)
    - Could improve with simple heuristics (maximize VP gain)
    - Or learned lightweight rollout policy

14. **PUCT Exploration Constant**
    - Current: cpuct=1.5 (AlphaGo default)
    - May need tuning for Catan's branching factor
    - Consider adaptive cpuct based on game phase

15. **Virtual Loss for Parallelization**
    - Not implemented yet
    - Needed for multi-threaded MCTS with shared tree
    - Prevents redundant exploration of same paths

### Game-Specific Features

16. **Development Card Strategy**
    - ✅ Knights, Road Building, Year of Plenty, Monopoly all playable
    - ✅ VP cards properly counted
    - MCTS may not value correctly initially - will improve with training
    - Need to model opponent reactions (e.g., Monopoly impact)

17. **Trading with Opponents**
    - Only bank/port trades implemented (sufficient for initial training)
    - Player-to-player trading could be huge strategic element
    - Complex negotiation space - can add later if agent plateaus

18. **Robber Placement Strategy**
    - Currently just blocks resources on 7
    - Stealing is implemented but strategy is complex
    - Could be modeled as separate subpolicy

19. **Initial Placement Strategy**
    - Setup phase is critical (determines starting resources)
    - May benefit from separate policy or hard-coded heuristics
    - Consider: diversity number balance, harbor access

### Training Infrastructure

20. **Replay Buffer Management**
    - Need efficient storage for training data
    - Consider: prioritized replay for critical positions
    - Disk vs memory tradeoff for large-scale training

21. **Curriculum Learning**
    - Start with 2-player, then 3-player, then 4-player?
    - Or start with simpler victory condition (5 VP instead of 10)?
    - Could help with sparse rewards

22. **Self-Play Diversity**
    - All players using same network → Nash equilibrium
    - Consider: past checkpoints, noise injection, play against random
    - Prevents overfitting to single strategy

23. **Evaluation Metrics**
    - Win rate vs baselines (random, pure MCTS, heuristic)
    - Elo rating system for tracking improvement
    - Separate metrics for different game phases

### Code Quality & Debugging

24. **Unit Tests for Game Logic**
    - Basic tests exist but could be more comprehensive
    - Test edge cases: robber interactions, dev card timing, etc.
    - Fuzzing with random game sequences

25. **MCTS Tree Visualization**
    - Would help debug policy learning
    - Show top actions, visit counts, Q-values
    - Could identify where policy/value disagree

26. **Profiling & Optimization**
    - Profile where time is spent (MCTS, NN, game logic)
    - Consider SIMD for state copying, legal move generation
    - Memory pooling for MCTS nodes (reduce allocations)

## Next Steps Priority

### Phase 1: Get AlphaZero Working (Current)
- [x] Pure MCTS baseline
- [x] AlphaZero MCTS structure
- [x] Complete game engine (longest road, resource bank, dev cards)
- [x] State encoder for NN input (~900 features)
- [x] Basic NN architecture (ResNet-style with policy + value heads)
- [x] Training loop (self-play → train → evaluate)
- [x] Integration test script
- [ ] Run initial training and verify learning

### Phase 2: Basic Training
- [ ] Implement proper state encoding
- [ ] Small NN (ResNet-like or Transformer)
- [ ] Train for 10k games, check if it beats random
- [ ] Train for 100k games, check if it beats pure MCTS

### Phase 3: Scale & Optimize
- [ ] Batched NN inference
- [ ] Multi-threaded MCTS
- [ ] Larger network, longer training
- [ ] Hyperparameter tuning

### Phase 4: Game-Specific Improvements
- [x] Implement longest road calculation
- [ ] Add player-to-player trading (optional)
- [ ] Fine-tune initial placement
- [ ] Separate robber subpolicy

## Performance Targets

- **Current (Pure MCTS)**: ~1.7s/game @ 100 iterations
- **With AlphaZero**: Target ~0.5-1.0s/game @ 200 simulations (NN replaces rollouts)
- **With Batching**: Target ~0.1-0.3s/game @ 400 simulations
- **Training Scale**: 1M games in ~72 hours (reasonable for initial training)

## Notes

- Keep pure MCTS as baseline - useful for ablation studies
- Document all hyperparameters for reproducibility
- Version control checkpoints with eval metrics
- Remember: Catan has high variance - need many games to see signal
