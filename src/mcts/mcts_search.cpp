// mcts_search.cpp
// Implementation of pure MCTS with random rollouts.

#include "mcts/mcts_search.h"
#include <algorithm>

namespace catan {
namespace mcts {

Action MCTSSearch::search(const GameState& root_state) {
    // Create root node
    root_ = std::make_unique<MCTSNode>();
    root_->player_idx = root_state.current_player;
    root_->root_player_idx = root_state.current_player;
    
    // Generate legal actions for root
    generate_legal_actions(root_state, root_->untried_actions);
    
    if (root_->untried_actions.empty()) {
        // No legal actions, return invalid action
        return Action{};
    }
    
    if (root_->untried_actions.size() == 1) {
        // Only one legal action, no need to search
        return root_->untried_actions[0];
    }
    
    // Run MCTS iterations
    for (std::uint32_t i = 0; i < config_.num_iterations; ++i) {
        // Copy state for this iteration
        GameState state = root_state;
        
        // 1. Selection: traverse tree using UCB1
        MCTSNode* node = select(root_.get(), state);
        
        // 2. Expansion: add new child node
        node = expand(node, state);
        
        // 3. Simulation: random rollout from expanded node
        double value = simulate(state, root_state.current_player);
        
        // 4. Backpropagation: update statistics up the tree
        backpropagate(node, value);
    }
    
    // Select best action based on configuration
    MCTSNode* best_child = config_.use_visit_count 
        ? root_->get_most_visited_child()
        : root_->get_best_value_child();
    
    if (best_child == nullptr) {
        // Fallback to first legal action
        return root_->untried_actions.empty() 
            ? root_->children[0]->action 
            : root_->untried_actions[0];
    }
    
    return best_child->action;
}

MCTSNode* MCTSSearch::select(MCTSNode* node, GameState& state) {
    // Traverse tree until we reach a leaf or an unexpanded node
    while (!node->is_terminal(state) && node->is_fully_expanded() && !node->children.empty()) {
        node = node->select_child(config_.exploration_constant);
        
        if (node == nullptr) {
            break; // Safety check
        }
        
        // Apply action to advance state
        apply_action(state, node->action, rng_());
    }
    
    return node;
}

MCTSNode* MCTSSearch::expand(MCTSNode* node, GameState& state) {
    // If node is terminal or has no untried actions, return as-is
    if (node->is_terminal(state) || node->untried_actions.empty()) {
        return node;
    }
    
    // Pick a random untried action
    std::uniform_int_distribution<std::size_t> dist(0, node->untried_actions.size() - 1);
    std::size_t idx = dist(rng_);
    
    Action action = node->untried_actions[idx];
    
    // Remove from untried list
    node->untried_actions.erase(node->untried_actions.begin() + idx);
    
    // Apply action to state
    apply_action(state, action, rng_());
    
    // Create child node
    auto child = std::make_unique<MCTSNode>();
    child->parent = node;
    child->action = action;
    child->player_idx = state.current_player;
    child->root_player_idx = node->root_player_idx;  // Inherit from parent
    
    // Generate legal actions for child
    generate_legal_actions(state, child->untried_actions);
    
    // Add to parent's children
    MCTSNode* child_ptr = child.get();
    node->children.push_back(std::move(child));
    
    return child_ptr;
}

double MCTSSearch::simulate(const GameState& state, std::uint8_t root_player_idx) {
    // Biased random rollout until game ends or max depth
    // Bias: prefer building actions and ending turns over trading
    GameState sim_state = state;
    std::uint32_t depth = 0;
    
    while (!sim_state.is_game_over() && depth < config_.max_rollout_depth) {
        // Get legal actions
        std::vector<Action> actions;
        generate_legal_actions(sim_state, actions);
        
        if (actions.empty()) {
            break; // Stuck state, should not happen
        }
        
        // Bias towards productive actions:
        // 1. Find end_turn action if present - 30% chance to take it
        // 2. Otherwise prefer building over trading
        Action selected_action;
        
        // Look for end_turn
        bool has_end_turn = false;
        for (const Action& a : actions) {
            if (a.type == ActionType::EndTurn) {
                has_end_turn = true;
                
                // 30% chance to end turn immediately (speeds up rollouts)
                std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
                if (prob_dist(rng_) < 0.3) {
                    selected_action = a;
                    break;
                }
            }
        }
        
        // If didn't select end_turn, pick action with bias
        if (selected_action.type != ActionType::EndTurn) {
            // Separate actions into building and other
            std::vector<Action> building_actions;
            std::vector<Action> other_actions;
            
            for (const Action& a : actions) {
                ActionType t = a.type;
                if (t == ActionType::PlaceSettlement || 
                    t == ActionType::PlaceRoad || 
                    t == ActionType::UpgradeToCity ||
                    t == ActionType::PlaceInitialSettlement ||
                    t == ActionType::PlaceInitialRoad) {
                    building_actions.push_back(a);
                } else {
                    other_actions.push_back(a);
                }
            }
            
            // 70% chance to pick building action if available
            std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
            if (!building_actions.empty() && prob_dist(rng_) < 0.7) {
                std::uniform_int_distribution<std::size_t> dist(0, building_actions.size() - 1);
                selected_action = building_actions[dist(rng_)];
            } else {
                std::uniform_int_distribution<std::size_t> dist(0, actions.size() - 1);
                selected_action = actions[dist(rng_)];
            }
        }
        
        // Apply action
        apply_action(sim_state, selected_action, rng_());
        
        depth++;
    }
    
    // Return reward from root player's perspective
    if (sim_state.is_game_over()) {
        std::uint8_t winner = sim_state.get_winner();
        if (winner == root_player_idx) {
            return 1.0; // Win
        } else if (winner != 0xFF) {
            return 0.0; // Loss
        }
    }
    
    // If max depth reached or draw, use heuristic (victory points)
    // Normalize to [0, 1] range
    std::uint8_t root_vp = sim_state.players[root_player_idx].total_victory_points();
    std::uint8_t max_vp = 0;
    for (std::uint8_t p = 0; p < sim_state.num_players; ++p) {
        std::uint8_t vp = sim_state.players[p].total_victory_points();
        if (vp > max_vp) max_vp = vp;
    }
    
    if (max_vp == 0) return 0.5; // No one has any points
    
    return static_cast<double>(root_vp) / static_cast<double>(max_vp);
}

void MCTSSearch::backpropagate(MCTSNode* node, double value) {
    // Propagate value up the tree
    // In multi-player games, value is ALWAYS from root player's perspective
    // No negation needed - this is NOT a zero-sum game
    while (node != nullptr) {
        node->visit_count++;
        node->value_sum += value;
        node = node->parent;
    }
}

} // namespace mcts
} // namespace catan
