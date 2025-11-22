# AlphaZero MCTS Optimization Plan

## Identified Bottlenecks (from profiling results)

### Current Performance: 0.83s/game with 25 MCTS sims
- **3,420 NN requests per game** = 136.8 requests per simulation
- **4.12 average batch size** (max 64) = only 6.4% GPU utilization
- **Batch efficiency at 58-90% scaling** (200 sims drops to 3.59 batch size)

## Critical Optimization Opportunities

### 1. **State Copying Overhead** ⚠️ HIGH IMPACT
**Location**: `alphazero_mcts.cpp:48` - Every simulation copies entire GameState
```cpp
for (std::uint32_t i = 0; i < config_.num_simulations; ++i) {
    GameState state = root_state;  // EXPENSIVE COPY
```
**Problem**: 25 full GameState copies per move, each containing vectors of players, board state, etc.
**Solution**: Virtual board + incremental state updates with undo stack
**Expected Speedup**: 20-30% (reduces allocation overhead)

### 2. **Synchronous NN Evaluation** ⚠️ CRITICAL
**Location**: `alphazero_mcts.cpp:103-104, 149-150` - Two separate NN calls per node
**Problem**: 
- `expand_and_evaluate()` calls NN twice for already-expanded nodes (lines 137-149)
- Every new node expansion requires NN call before MCTS continues
- GPU sits idle between requests (poor batching)
**Solution**: 
- Cache NN results in nodes
- Virtual loss to allow parallel tree traversal
- Batch multiple simulations' NN requests together
**Expected Speedup**: 40-60% (increases batch size from 4.12 to ~20-40)

### 3. **Child State Prediction** 🔧 MEDIUM IMPACT
**Location**: `alphazero_mcts.cpp:183-186` - Creates child GameState just to get player_idx
```cpp
GameState child_state = state;
apply_action(child_state, child->action, rng_());
child->player_idx = child_state.current_player;
```
**Problem**: Expensive state copy + action application just to determine next player
**Solution**: Lightweight function to predict next player from action type
**Expected Speedup**: 5-10%

### 4. **Prior Finding Loop** 🔧 LOW-MEDIUM IMPACT
**Location**: `alphazero_mcts.h:99-106` - O(n) search in get_puct_score()
```cpp
for (std::size_t i = 0; i < parent->children.size(); ++i) {
    if (parent->children[i].get() == this) {
        prior = parent->prior_probs[i];
        break;
    }
}
```
**Problem**: Every PUCT calculation searches through siblings
**Solution**: Store action_index in node or cache prior probability
**Expected Speedup**: 3-5%

### 5. **Worker Count** ⚡ FREE SPEEDUP
**Current**: 12 workers
**Available**: 16 CPU cores
**Expected Speedup**: 10-15% (more parallel games = better batching)

## Implementation Priority

### Phase 1: Quick Wins (< 30 minutes)
- [x] Increase worker count to 16
- [ ] Disable checkpointing for benchmarks
- [ ] Add --no-save flag to train_alphazero.py
- [ ] Cache prior in node (avoid O(n) search)

### Phase 2: Virtual Loss + Batching (1-2 hours)
- [ ] Add virtual loss to enable parallel tree descent
- [ ] Batch multiple simulations' NN requests
- [ ] Increase target batch size from 4 to 32+
- [ ] Expected: 0.83s → ~0.35s per game

### Phase 3: State Management (2-3 hours)
- [ ] Implement incremental state updates with undo
- [ ] Eliminate GameState copies in simulation loop
- [ ] Expected: Additional 20% speedup

### Phase 4: Advanced Optimizations (if needed)
- [ ] Cache NN evaluations for repeated positions
- [ ] Use FEN-style hashing for transposition table
- [ ] SIMD optimizations for PUCT calculations

## Target Performance
- **Current**: 0.83s/game (25 sims) = 9.6 days for 1M games
- **After Phase 1**: 0.70s/game = 8.1 days
- **After Phase 2**: 0.35s/game = 4.0 days ⭐ TARGET
- **After Phase 3**: 0.28s/game = 3.2 days

## Testing Strategy
1. Modify single_game.py to add detailed timing
2. Test each optimization in isolation
3. Rebuild C++ extension on GPU node
4. Run 10-game benchmark after each change
5. Keep optimizations that show >5% improvement
