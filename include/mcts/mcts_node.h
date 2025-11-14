// mcts_node.h
// Pure MCTS tree node for Catan self-play.
// Uses UCB1 for selection and random rollouts for evaluation.

#pragma once

#include "../game_state.h"
#include "../action.h"
#include <vector>
#include <memory>
#include <cmath>
#include <limits>

namespace catan {
namespace mcts {

// MCTS tree node
// Each node represents a game state and tracks visit statistics
struct MCTSNode {
    // Parent node (nullptr for root)
    MCTSNode* parent{nullptr};
    
    // Action taken from parent to reach this node
    Action action{};
    
    // Children nodes (expanded lazily)
    std::vector<std::unique_ptr<MCTSNode>> children{};
    
    // Legal actions from this state (computed once on expansion)
    std::vector<Action> untried_actions{};
    
    // Statistics
    std::uint32_t visit_count{0};
    double value_sum{0.0};  // Sum of rewards from root player's perspective
    
    // Player who made the move to reach this node
    std::uint8_t player_idx{0};
    
    // Root player (whose perspective all values are from)
    std::uint8_t root_player_idx{0};
    
    // Check if node is fully expanded (all actions have been tried)
    bool is_fully_expanded() const {
        return untried_actions.empty();
    }
    
    // Check if node is terminal (game over)
    bool is_terminal(const GameState& state) const {
        return state.is_game_over();
    }
    
    // Get average value (win rate)
    double get_value() const {
        if (visit_count == 0) return 0.0;
        return value_sum / static_cast<double>(visit_count);
    }
    
    // UCB1 score for selection
    // Higher score = more promising to explore
    double get_ucb_score(double exploration_constant = 1.41) const {
        if (visit_count == 0) {
            return std::numeric_limits<double>::infinity();
        }
        
        if (parent == nullptr || parent->visit_count == 0) {
            return get_value();
        }
        
        // UCB1: exploitation + exploration
        double exploitation = get_value();
        double exploration = exploration_constant * 
            std::sqrt(std::log(static_cast<double>(parent->visit_count)) / 
                      static_cast<double>(visit_count));
        
        return exploitation + exploration;
    }
    
    // Select best child using UCB1
    MCTSNode* select_child(double exploration_constant = 1.41) {
        MCTSNode* best_child = nullptr;
        double best_score = -std::numeric_limits<double>::infinity();
        
        for (auto& child : children) {
            double score = child->get_ucb_score(exploration_constant);
            if (score > best_score) {
                best_score = score;
                best_child = child.get();
            }
        }
        
        return best_child;
    }
    
    // Get most visited child (for final action selection)
    MCTSNode* get_most_visited_child() const {
        MCTSNode* best_child = nullptr;
        std::uint32_t max_visits = 0;
        
        for (const auto& child : children) {
            if (child->visit_count > max_visits) {
                max_visits = child->visit_count;
                best_child = child.get();
            }
        }
        
        return best_child;
    }
    
    // Get child with highest value (alternative selection strategy)
    MCTSNode* get_best_value_child() const {
        MCTSNode* best_child = nullptr;
        double best_value = -std::numeric_limits<double>::infinity();
        
        for (const auto& child : children) {
            if (child->visit_count > 0) {
                double value = child->get_value();
                if (value > best_value) {
                    best_value = value;
                    best_child = child.get();
                }
            }
        }
        
        return best_child;
    }
};

} // namespace mcts
} // namespace catan
