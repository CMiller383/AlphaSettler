// alphazero_mcts.h
// AlphaZero-style MCTS with neural network guidance.
// Uses PUCT algorithm and NN policy/value instead of random rollouts.

#pragma once

#include "../game_state.h"
#include "../action.h"
#include "../move_gen.h"
#include "../state_transition.h"
#include <random>
#include <memory>
#include <vector>
#include <functional>

namespace catan {
namespace alphazero {

// Neural network evaluation result
struct NNEvaluation {
    std::vector<float> policy;  // Prior probabilities for each legal action
    float value;                 // Expected value from current player's perspective [-1, 1]
};

// Callback for neural network inference
// Takes game state, returns policy and value
using NNEvaluator = std::function<NNEvaluation(const GameState&)>;

// AlphaZero MCTS configuration
struct AlphaZeroConfig {
    std::uint32_t num_simulations{800};     // Number of MCTS simulations per move
    float cpuct{1.5f};                      // PUCT exploration constant (higher = more exploration)
    float dirichlet_alpha{0.3f};            // Dirichlet noise alpha for exploration at root
    float dirichlet_weight{0.25f};          // Weight of Dirichlet noise at root
    bool add_exploration_noise{true};       // Add noise to root for exploration
    std::uint32_t random_seed{0};           // Seed for randomness (0 = time-based)
};

// AlphaZero MCTS node
struct AlphaZeroNode {
    // Parent node (nullptr for root)
    AlphaZeroNode* parent{nullptr};
    
    // Action taken from parent to reach this node
    Action action{};
    
    // Children nodes (expanded lazily)
    std::vector<std::unique_ptr<AlphaZeroNode>> children{};
    
    // Legal actions and their prior probabilities from NN
    std::vector<Action> legal_actions{};
    std::vector<float> prior_probs{};
    
    // Statistics
    std::uint32_t visit_count{0};
    float value_sum{0.0f};      // Sum of values from current player's perspective
    
    // Player who made the move to reach this node
    std::uint8_t player_idx{0};
    
    // Root player (whose perspective all values are from)
    std::uint8_t root_player_idx{0};
    
    // Check if node is fully expanded
    bool is_fully_expanded() const {
        return !legal_actions.empty() && children.size() == legal_actions.size();
    }
    
    // Check if node is terminal
    bool is_terminal(const GameState& state) const {
        return state.is_game_over();
    }
    
    // Get average value (Q-value)
    float get_q_value() const {
        if (visit_count == 0) return 0.0f;
        return value_sum / static_cast<float>(visit_count);
    }
    
    // PUCT score for child selection
    // U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    float get_puct_score(float cpuct, float parent_visit_sqrt) const {
        float q_value = get_q_value();
        
        // Find this node's index in parent's children to get prior
        if (parent == nullptr) return q_value;
        
        float prior = 0.0f;
        for (std::size_t i = 0; i < parent->children.size(); ++i) {
            if (parent->children[i].get() == this) {
                prior = parent->prior_probs[i];
                break;
            }
        }
        
        // PUCT formula
        float exploration = cpuct * prior * parent_visit_sqrt / (1.0f + static_cast<float>(visit_count));
        
        return q_value + exploration;
    }
    
    // Select best child using PUCT
    AlphaZeroNode* select_child(float cpuct) {
        if (children.empty()) return nullptr;
        
        float parent_visit_sqrt = std::sqrt(static_cast<float>(visit_count));
        
        AlphaZeroNode* best_child = nullptr;
        float best_score = -std::numeric_limits<float>::infinity();
        
        for (auto& child : children) {
            float score = child->get_puct_score(cpuct, parent_visit_sqrt);
            if (score > best_score) {
                best_score = score;
                best_child = child.get();
            }
        }
        
        return best_child;
    }
    
    // Get most visited child (for final action selection)
    AlphaZeroNode* get_most_visited_child() const {
        AlphaZeroNode* best_child = nullptr;
        std::uint32_t max_visits = 0;
        
        for (const auto& child : children) {
            if (child->visit_count > max_visits) {
                max_visits = child->visit_count;
                best_child = child.get();
            }
        }
        
        return best_child;
    }
};

// AlphaZero MCTS search engine
class AlphaZeroMCTS {
public:
    AlphaZeroMCTS(const AlphaZeroConfig& config, NNEvaluator evaluator)
        : config_(config)
        , evaluator_(std::move(evaluator))
        , rng_(config.random_seed == 0 ? std::random_device{}() : config.random_seed)
    {}
    
    // Run MCTS search and return best action
    Action search(const GameState& root_state);
    
    // Get action probabilities from visit counts (for training)
    std::vector<float> get_action_probabilities() const;
    
    // Get root node (for analysis)
    const AlphaZeroNode* get_root() const { return root_.get(); }
    
private:
    // MCTS phases
    AlphaZeroNode* select(AlphaZeroNode* node, GameState& state);
    void expand_and_evaluate(AlphaZeroNode* node, GameState& state);
    void backpropagate(AlphaZeroNode* node, float value);
    
    // Add Dirichlet noise to root for exploration
    void add_exploration_noise(AlphaZeroNode* root);
    
    // Configuration
    AlphaZeroConfig config_;
    
    // Neural network evaluator
    NNEvaluator evaluator_;
    
    // Random number generator
    std::mt19937 rng_;
    
    // Root node (cleared each search)
    std::unique_ptr<AlphaZeroNode> root_;
};

} // namespace alphazero
} // namespace catan
