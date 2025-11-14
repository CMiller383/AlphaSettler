// mcts_agent.h
// High-level MCTS agent interface for self-play and evaluation.

#pragma once

#include "mcts_search.h"
#include "../game_state.h"
#include "../action.h"
#include <vector>
#include <utility>

namespace catan {
namespace mcts {

// Action with visit-based probability (for training data collection)
struct ActionPolicy {
    Action action;
    double probability;  // Normalized visit count
    std::uint32_t visits;
};

// MCTS agent that plays Catan
class MCTSAgent {
public:
    explicit MCTSAgent(const MCTSConfig& config = MCTSConfig{})
        : config_(config)
        , search_(config)
    {}
    
    // Select action from current game state
    Action select_action(const GameState& state);
    
    // Get action policy (visit counts normalized to probabilities)
    // Useful for generating training data for neural networks
    std::vector<ActionPolicy> get_action_policy(const GameState& state);
    
    // Play full game and return winner
    // Returns 0xFF if game doesn't finish within max_actions
    std::uint8_t play_game(GameState& state, std::uint32_t max_actions = 500);
    
    // Get configuration
    const MCTSConfig& get_config() const { return config_; }
    
    // Update configuration (takes effect on next search)
    void set_config(const MCTSConfig& config) { 
        config_ = config;
        search_ = MCTSSearch(config);
    }
    
private:
    MCTSConfig config_;
    MCTSSearch search_;
};

// Run multiple self-play games and collect statistics
struct SelfPlayStats {
    std::uint32_t games_played{0};
    std::uint32_t games_finished{0};
    std::array<std::uint32_t, MAX_PLAYERS> wins_by_player{};
    double avg_game_length{0.0};
    double avg_final_vp{0.0};
};

// Self-play engine for data generation
class SelfPlayEngine {
public:
    explicit SelfPlayEngine(const MCTSConfig& config = MCTSConfig{})
        : agent_(config)
    {}
    
    // Play N games and collect statistics
    SelfPlayStats play_games(std::uint32_t num_games, 
                             std::uint32_t num_players = 4,
                             std::uint32_t seed = 0);
    
    // Get agent configuration
    const MCTSConfig& get_config() const { return agent_.get_config(); }
    
    // Update agent configuration
    void set_config(const MCTSConfig& config) { agent_.set_config(config); }
    
private:
    MCTSAgent agent_;
};

} // namespace mcts
} // namespace catan
