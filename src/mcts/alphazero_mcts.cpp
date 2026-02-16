// alphazero_mcts.cpp
// Implementation of AlphaZero-style MCTS with neural network guidance.

#include "mcts/alphazero_mcts.h"
#include "batched_evaluator.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

namespace catan {
namespace alphazero {

namespace {
    // Convert value from one player's perspective to another's in multi-player game
    // For 4-player Catan, this is complex because it's not zero-sum
    // We use a position-based heuristic: opponent values are negatively correlated
    inline float convert_value_perspective(float value, std::uint8_t from_player, std::uint8_t to_player, std::uint8_t num_players) {
        if (from_player == to_player) {
            return value;  // Same player, no conversion needed
        }
        
        // In multi-player games, we approximate opponent perspective
        // Key insight: if it's good for them, it's likely bad for us
        // Use a weighted negation based on number of players
        // For 4 players: value is divided among opponents
        // If player A has value +0.8, each opponent has roughly -0.8/3 ≈ -0.27
        float opponent_factor = -1.0f / static_cast<float>(num_players - 1);
        return value * opponent_factor;
    }
}

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
    
    // Run MCTS simulations in parallel batches for better GPU utilization
    std::uint32_t num_batches = (config_.num_simulations + config_.num_parallel_sims - 1) / config_.num_parallel_sims;
    
    for (std::uint32_t batch = 0; batch < num_batches; ++batch) {
        std::uint32_t batch_size = std::min(config_.num_parallel_sims, 
                                            config_.num_simulations - batch * config_.num_parallel_sims);
        
        // Store simulation paths and states for this batch
        std::vector<std::pair<AlphaZeroNode*, GameState>> sim_paths;
        sim_paths.reserve(batch_size);
        
        // Phase 1: Selection with virtual loss - collect all leaf nodes
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            GameState state = root_state;
            AlphaZeroNode* node = select_with_virtual_loss(root_.get(), state);
            sim_paths.emplace_back(node, state);
        }
        
        // Phase 2: Prepare nodes and collect those needing NN evaluation
        std::vector<std::pair<AlphaZeroNode*, GameState*>> nodes_to_evaluate;
        nodes_to_evaluate.reserve(batch_size);
        
        for (auto& [node, state] : sim_paths) {
            bool needs_eval = prepare_for_evaluation(node, state);
            if (needs_eval) {
                nodes_to_evaluate.emplace_back(node, &state);
            }
        }
        
        // Phase 3: Evaluate all nodes
        std::vector<NNEvaluation> evaluations;
        evaluations.reserve(nodes_to_evaluate.size());
        
        if (batched_evaluator_ && nodes_to_evaluate.size() > 1) {
            // Use batched evaluation for multiple nodes
            std::vector<const GameState*> states;
            std::vector<std::uint8_t> players;
            std::vector<std::size_t> num_actions;
            
            states.reserve(nodes_to_evaluate.size());
            players.reserve(nodes_to_evaluate.size());
            num_actions.reserve(nodes_to_evaluate.size());
            
            for (auto& [node, state_ptr] : nodes_to_evaluate) {
                states.push_back(state_ptr);
                players.push_back(state_ptr->current_player);
                num_actions.push_back(node->legal_actions.size());
            }
            
            // Call batch evaluation
            auto batch_results = batched_evaluator_->evaluate_batch(states, players, num_actions);
            
            // Convert to NNEvaluation format
            for (auto& [policy, value] : batch_results) {
                NNEvaluation eval;
                eval.policy = std::move(policy);
                eval.value = value;
                evaluations.push_back(std::move(eval));
            }
        } else {
            // Fall back to sequential evaluation
            for (auto& [node, state_ptr] : nodes_to_evaluate) {
                evaluations.push_back(evaluator_(*state_ptr));
            }
        }
        
        // Phase 4: Apply evaluation results to nodes
        for (std::size_t i = 0; i < nodes_to_evaluate.size(); ++i) {
            apply_evaluation(nodes_to_evaluate[i].first, evaluations[i], *nodes_to_evaluate[i].second);
        }
        
        // Phase 5: Backpropagation and remove virtual loss
        for (auto& [node, state] : sim_paths) {
            float value = node->get_q_value();
            backpropagate_and_remove_virtual_loss(node, value);
        }
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

AlphaZeroNode* AlphaZeroMCTS::select_with_virtual_loss(AlphaZeroNode* node, GameState& state) {
    // Traverse tree until we reach a leaf node, applying virtual loss along the path
    while (!node->is_terminal(state) && node->is_fully_expanded()) {
        // Select child using PUCT with virtual loss penalty
        node = node->select_child(config_.cpuct, config_.virtual_loss_penalty);
        
        if (node == nullptr) {
            break; // Safety
        }
        
        // Apply virtual loss to discourage other threads from selecting this node
        node->virtual_loss.fetch_add(1, std::memory_order_relaxed);
        
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
    
    // CRITICAL FIX: Node should NEVER have children here!
    // If selection is working correctly, we only reach unexpanded leaves.
    // Re-evaluating already-expanded nodes is a bug that causes shallow search.
    if (!node->children.empty()) {
        // This should never happen - log error and handle gracefully
        // Just increment visit count and return (value already set from previous expansion)
        node->visit_count++;
        return;
    }
    
    // First visit - expand this node
    // Generate legal actions if not already done
    if (node->legal_actions.empty()) {
        generate_legal_actions(state, node->legal_actions);
    }
    
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
        child->cached_prior = node->prior_probs[i];  // Cache prior to avoid search later
        
        // Apply action to determine child's player
        GameState child_state = state;
        apply_action(child_state, child->action, rng_());
        child->player_idx = child_state.current_player;
        
        node->children.push_back(std::move(child));
    }
    
    // Update node with NN value (convert to root player's perspective)
    float value = convert_value_perspective(eval.value, state.current_player, node->root_player_idx, state.num_players);
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

void AlphaZeroMCTS::backpropagate_and_remove_virtual_loss(AlphaZeroNode* node, float value) {
    // Propagate value and remove virtual loss up the tree
    // Note: Start from node->parent since the leaf node doesn't have virtual loss applied
    AlphaZeroNode* current = node->parent;
    
    // Update leaf node statistics
    if (node != nullptr) {
        node->visit_count++;
        node->value_sum += value;
    }
    
    // Propagate up, removing virtual loss from each node
    while (current != nullptr) {
        current->visit_count++;
        current->value_sum += value;
        
        // Remove virtual loss (it was added during selection)
        int prev_vl = current->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
        
        // Safety check: virtual loss should never go negative
        if (prev_vl <= 0) {
            // This shouldn't happen, but handle gracefully
            current->virtual_loss.store(0, std::memory_order_relaxed);
        }
        
        current = current->parent;
    }
}

bool AlphaZeroMCTS::prepare_for_evaluation(AlphaZeroNode* node, GameState& state) {
    // Terminal node - set value directly, no NN needed
    if (node->is_terminal(state)) {
        std::uint8_t winner = state.get_winner();
        float value = (winner == node->root_player_idx) ? 1.0f : 0.0f;
        node->value_sum = value;
        node->visit_count = 1;
        return false;  // No NN evaluation needed
    }
    
    // Already expanded node - needs re-evaluation
    if (!node->children.empty()) {
        // Will need NN evaluation for value update
        return true;
    }
    
    // First visit - generate legal actions
    if (node->legal_actions.empty()) {
        generate_legal_actions(state, node->legal_actions);
    }
    
    if (node->legal_actions.empty()) {
        // Stuck state - no NN needed
        node->visit_count = 1;
        return false;
    }
    
    // Node needs expansion - requires NN evaluation
    return true;
}

void AlphaZeroMCTS::apply_evaluation(AlphaZeroNode* node, const NNEvaluation& eval, GameState& state) {
    // This is called after NN evaluation completes
    
    // If already expanded, just update value
    if (!node->children.empty()) {
        float value = convert_value_perspective(eval.value, state.current_player, node->root_player_idx, state.num_players);
        node->value_sum += value;
        node->visit_count++;
        return;
    }
    
    // First expansion - store priors and create children
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
        child->cached_prior = node->prior_probs[i];
        
        // Apply action to determine child's player
        GameState child_state = state;
        apply_action(child_state, child->action, rng_());
        child->player_idx = child_state.current_player;
        
        node->children.push_back(std::move(child));
    }
    
    // Update node with NN value (convert to root player's perspective)
    float value = convert_value_perspective(eval.value, state.current_player, node->root_player_idx, state.num_players);
    node->value_sum += value;
    node->visit_count++;
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
