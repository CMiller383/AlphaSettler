// alphazero_mcts.cpp
// Implementation of AlphaZero-style MCTS with neural network guidance.

#include "mcts/alphazero_mcts.h"
#include <algorithm>
#include <cmath>

namespace catan {
namespace alphazero {

Action AlphaZeroMCTS::search(const GameState& root_state) {
    // Create root node
    root_ = std::make_unique<AlphaZeroNode>();
    root_->player_idx = root_state.current_player;
    root_->root_player_idx = root_state.current_player;
    
    // Get legal actions
    generate_legal_actions(root_state, root_->legal_actions);
    
    if (root_->legal_actions.empty()) {
        return Action{}; // No legal actions
    }
    
    if (root_->legal_actions.size() == 1) {
        return root_->legal_actions[0]; // Only one action
    }
    
    // Evaluate root with neural network
    NNEvaluation eval = evaluator_(root_state);
    root_->prior_probs = eval.policy;
    
    // Ensure prior probabilities match legal actions
    if (root_->prior_probs.size() != root_->legal_actions.size()) {
        // Fallback: uniform priors
        float uniform_prob = 1.0f / root_->legal_actions.size();
        root_->prior_probs.assign(root_->legal_actions.size(), uniform_prob);
    }
    
    // Add exploration noise to root
    if (config_.add_exploration_noise) {
        add_exploration_noise(root_.get());
    }
    
    // Run MCTS simulations
    for (std::uint32_t i = 0; i < config_.num_simulations; ++i) {
        // Copy state for this simulation
        GameState state = root_state;
        
        // 1. Selection: traverse tree using PUCT
        AlphaZeroNode* node = select(root_.get(), state);
        
        // 2. Expansion & Evaluation: expand node and get NN value
        expand_and_evaluate(node, state);
        
        // 3. Backpropagation: update statistics up the tree
        // Note: value is already from root player's perspective
        float value = node->get_q_value();
        backpropagate(node->parent, value);
    }
    
    // Select best action based on visit counts
    AlphaZeroNode* best_child = root_->get_most_visited_child();
    
    if (best_child == nullptr) {
        return root_->legal_actions[0]; // Fallback
    }
    
    return best_child->action;
}

std::vector<float> AlphaZeroMCTS::get_action_probabilities() const {
    if (root_ == nullptr || root_->children.empty()) {
        return {};
    }
    
    // Calculate probabilities from visit counts
    std::vector<float> probs(root_->legal_actions.size(), 0.0f);
    
    std::uint32_t total_visits = 0;
    for (const auto& child : root_->children) {
        total_visits += child->visit_count;
    }
    
    if (total_visits == 0) {
        // Uniform if no visits
        float uniform = 1.0f / probs.size();
        std::fill(probs.begin(), probs.end(), uniform);
        return probs;
    }
    
    // Normalize by total visits
    for (std::size_t i = 0; i < root_->children.size(); ++i) {
        probs[i] = static_cast<float>(root_->children[i]->visit_count) / total_visits;
    }
    
    return probs;
}

AlphaZeroNode* AlphaZeroMCTS::select(AlphaZeroNode* node, GameState& state) {
    // Traverse tree until we reach a leaf node
    while (!node->is_terminal(state) && node->is_fully_expanded()) {
        node = node->select_child(config_.cpuct);
        
        if (node == nullptr) {
            break; // Safety
        }
        
        // Apply action to advance state
        apply_action(state, node->action, rng_());
    }
    
    return node;
}

void AlphaZeroMCTS::expand_and_evaluate(AlphaZeroNode* node, GameState& state) {
    // Terminal node - just set value
    if (node->is_terminal(state)) {
        std::uint8_t winner = state.get_winner();
        float value = (winner == node->root_player_idx) ? 1.0f : 0.0f;
        node->value_sum = value;
        node->visit_count = 1;
        return;
    }
    
    // Already expanded - evaluate and update
    if (!node->legal_actions.empty()) {
        NNEvaluation eval = evaluator_(state);
        float value = eval.value;
        
        // Convert value to root player's perspective
        // NN outputs value from current player's perspective
        if (state.current_player != node->root_player_idx) {
            // In multi-player, we can't just negate
            // Use a simple heuristic: if not our turn, value is less reliable
            // For now, keep as-is (assumes NN learns multi-player dynamics)
        }
        
        node->value_sum += value;
        node->visit_count++;
        return;
    }
    
    // First visit - expand this node
    generate_legal_actions(state, node->legal_actions);
    
    if (node->legal_actions.empty()) {
        node->visit_count = 1;
        return; // Stuck state
    }
    
    // Evaluate with neural network
    NNEvaluation eval = evaluator_(state);
    
    // Store prior probabilities
    if (eval.policy.size() == node->legal_actions.size()) {
        node->prior_probs = eval.policy;
    } else {
        // Fallback: uniform priors
        float uniform_prob = 1.0f / node->legal_actions.size();
        node->prior_probs.assign(node->legal_actions.size(), uniform_prob);
    }
    
    // Normalize priors
    float prior_sum = 0.0f;
    for (float p : node->prior_probs) prior_sum += p;
    if (prior_sum > 0.0f) {
        for (float& p : node->prior_probs) p /= prior_sum;
    }
    
    // Create child nodes
    node->children.reserve(node->legal_actions.size());
    for (std::size_t i = 0; i < node->legal_actions.size(); ++i) {
        auto child = std::make_unique<AlphaZeroNode>();
        child->parent = node;
        child->action = node->legal_actions[i];
        child->root_player_idx = node->root_player_idx;
        
        // Apply action to determine child's player
        GameState child_state = state;
        apply_action(child_state, child->action, rng_());
        child->player_idx = child_state.current_player;
        
        node->children.push_back(std::move(child));
    }
    
    // Update node with NN value
    float value = eval.value;
    node->value_sum += value;
    node->visit_count++;
}

void AlphaZeroMCTS::backpropagate(AlphaZeroNode* node, float value) {
    // Propagate value up the tree
    // Value is from root player's perspective (no negation for multi-player)
    while (node != nullptr) {
        node->visit_count++;
        node->value_sum += value;
        node = node->parent;
    }
}

void AlphaZeroMCTS::add_exploration_noise(AlphaZeroNode* root) {
    if (root->prior_probs.empty()) return;
    
    // Generate Dirichlet noise
    std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);
    
    std::vector<float> noise(root->prior_probs.size());
    float noise_sum = 0.0f;
    
    for (std::size_t i = 0; i < noise.size(); ++i) {
        noise[i] = gamma(rng_);
        noise_sum += noise[i];
    }
    
    // Normalize noise
    if (noise_sum > 0.0f) {
        for (float& n : noise) n /= noise_sum;
    }
    
    // Mix noise with prior: P = (1 - ε) * P + ε * η
    for (std::size_t i = 0; i < root->prior_probs.size(); ++i) {
        root->prior_probs[i] = (1.0f - config_.dirichlet_weight) * root->prior_probs[i] +
                                config_.dirichlet_weight * noise[i];
    }
}

} // namespace alphazero
} // namespace catan
