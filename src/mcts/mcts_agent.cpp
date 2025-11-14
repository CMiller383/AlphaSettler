// mcts_agent.cpp
// Implementation of MCTS agent and self-play engine.

#include "mcts/mcts_agent.h"
#include "state_transition.h"
#include "move_gen.h"
#include <random>

namespace catan {
namespace mcts {

Action MCTSAgent::select_action(const GameState& state) {
    return search_.search(state);
}

std::vector<ActionPolicy> MCTSAgent::get_action_policy(const GameState& state) {
    // Run MCTS search
    Action best_action = search_.search(state);
    
    // Get root node
    const MCTSNode* root = search_.get_root();
    if (root == nullptr || root->children.empty()) {
        // Fallback: single action with 100% probability
        return {{best_action, 1.0, 1}};
    }
    
    // Calculate total visits
    std::uint32_t total_visits = 0;
    for (const auto& child : root->children) {
        total_visits += child->visit_count;
    }
    
    if (total_visits == 0) {
        return {{best_action, 1.0, 1}};
    }
    
    // Build policy from visit counts
    std::vector<ActionPolicy> policy;
    policy.reserve(root->children.size());
    
    for (const auto& child : root->children) {
        ActionPolicy ap;
        ap.action = child->action;
        ap.visits = child->visit_count;
        ap.probability = static_cast<double>(child->visit_count) / 
                        static_cast<double>(total_visits);
        policy.push_back(ap);
    }
    
    return policy;
}

std::uint8_t MCTSAgent::play_game(GameState& state, std::uint32_t max_actions) {
    std::uint32_t actions_taken = 0;
    
    // Use a simple RNG for action application
    std::mt19937 rng(config_.random_seed);
    
    while (!state.is_game_over() && actions_taken < max_actions) {
        // Get legal actions first (sanity check)
        std::vector<Action> legal_actions;
        generate_legal_actions(state, legal_actions);
        
        if (legal_actions.empty()) {
            break; // Stuck state
        }
        
        // Select action using MCTS
        Action action = select_action(state);
        
        // Apply action
        apply_action(state, action, rng());
        
        actions_taken++;
    }
    
    return state.get_winner();
}

SelfPlayStats SelfPlayEngine::play_games(std::uint32_t num_games, 
                                         std::uint32_t num_players,
                                         std::uint32_t seed) {
    SelfPlayStats stats;
    stats.games_played = num_games;
    
    std::uint32_t total_actions = 0;
    std::uint32_t total_vp = 0;
    
    for (std::uint32_t i = 0; i < num_games; ++i) {
        // Create new game with varied seed
        GameState game = GameState::create_new_game(num_players, seed + i);
        
        // Play game
        std::uint8_t winner = agent_.play_game(game, 10000);
        
        // Collect statistics
        if (winner != 0xFF) {
            stats.games_finished++;
            if (winner < MAX_PLAYERS) {
                stats.wins_by_player[winner]++;
            }
        }
        
        // Count actions (estimate from turn number)
        total_actions += game.turn_number;
        
        // Sum victory points
        for (std::uint8_t p = 0; p < num_players; ++p) {
            total_vp += game.players[p].total_victory_points();
        }
    }
    
    // Calculate averages
    if (stats.games_played > 0) {
        stats.avg_game_length = static_cast<double>(total_actions) / 
                               static_cast<double>(stats.games_played);
        stats.avg_final_vp = static_cast<double>(total_vp) / 
                            static_cast<double>(stats.games_played * num_players);
    }
    
    return stats;
}

} // namespace mcts
} // namespace catan
