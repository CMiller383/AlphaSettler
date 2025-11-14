// mcts_search.h
// Pure MCTS search implementation for Catan.
// Uses random rollouts for state evaluation (no neural network).

#pragma once

#include "mcts_node.h"
#include "../game_state.h"
#include "../action.h"
#include "../move_gen.h"
#include "../state_transition.h"
#include <random>
#include <memory>

namespace catan {
namespace mcts {

// MCTS search configuration
struct MCTSConfig {
    std::uint32_t num_iterations{1000};     // Number of MCTS iterations per move
    double exploration_constant{1.41};      // UCB1 exploration parameter (sqrt(2) default)
    std::uint32_t max_rollout_depth{100};   // Maximum rollout depth before giving up
    std::uint32_t random_seed{0};           // Seed for randomness (0 = time-based)
    bool use_visit_count{true};             // True: select most visited, False: select best value
};

// Pure MCTS search engine
class MCTSSearch {
public:
    explicit MCTSSearch(const MCTSConfig& config = MCTSConfig{})
        : config_(config)
        , rng_(config.random_seed == 0 ? std::random_device{}() : config.random_seed)
    {}
    
    // Run MCTS search and return best action
    Action search(const GameState& root_state);
    
    // Get root node statistics (for debugging/analysis)
    const MCTSNode* get_root() const { return root_.get(); }
    
private:
    // MCTS phases
    MCTSNode* select(MCTSNode* node, GameState& state);
    MCTSNode* expand(MCTSNode* node, GameState& state);
    double simulate(const GameState& state, std::uint8_t player_idx);
    void backpropagate(MCTSNode* node, double value);
    
    // Helper: apply action and return child node
    MCTSNode* apply_action_and_get_child(MCTSNode* parent, const Action& action, GameState& state);
    
    // Configuration
    MCTSConfig config_;
    
    // Random number generator
    std::mt19937 rng_;
    
    // Root node (cleared each search)
    std::unique_ptr<MCTSNode> root_;
};

} // namespace mcts
} // namespace catan
