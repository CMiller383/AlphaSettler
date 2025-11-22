// batched_evaluator.h
// Thread-safe batched neural network evaluator for parallel self-play.
// Collects evaluation requests from multiple threads and processes them in batches.

#pragma once

#include "game_state.h"
#include "state_encoder.h"
#include "move_gen.h"
#include "action.h"
#include "mcts/alphazero_mcts.h"
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <atomic>
#include <thread>
#include <unordered_map>

namespace catan {

// Evaluation request from a worker thread
struct EvaluationRequest {
    GameState state;
    std::uint8_t perspective_player;
    std::uint32_t request_id;  // For matching responses
};

// Evaluation response to a worker thread
struct EvaluationResponse {
    std::uint32_t request_id;
    std::vector<float> policy;  // Prior probabilities for legal actions
    float value;                 // Expected value [-1, 1]
};

// Batch evaluation callback: takes multiple states, returns multiple evaluations
// OPTIMIZED: Input is pre-stacked 2D array to avoid Python list conversion overhead
// Input: (stacked_states_flat, num_legal_actions, batch_size, feature_size)
//   - stacked_states_flat: flat array of size (batch_size * feature_size)
//   - num_legal_actions: vector of size batch_size
// Output: vector of (policy, value) pairs in same order
using BatchEvaluatorCallback = std::function<
    std::vector<std::pair<std::vector<float>, float>>(
        const std::vector<float>&,           // stacked states (flat)
        const std::vector<std::size_t>&,     // num_legal_actions
        std::size_t,                         // batch_size
        std::size_t                          // feature_size
    )
>;

// Configuration for batched evaluator
struct BatchedEvaluatorConfig {
    std::size_t max_batch_size{32};     // Maximum batch size for NN inference
    std::size_t min_batch_size{1};      // Minimum batch size (wait for this many)
    std::uint32_t timeout_ms{10};       // Timeout for batch collection (milliseconds)
    bool enable_batching{true};         // If false, process immediately (for debugging)
};

// Thread-safe batched evaluator
// Collects evaluation requests from multiple threads and processes them in batches
class BatchedEvaluator {
public:
    BatchedEvaluator(
        const BatchedEvaluatorConfig& config,
        BatchEvaluatorCallback callback
    );
    
    ~BatchedEvaluator();
    
    // Submit evaluation request and wait for result
    // Thread-safe: can be called from multiple worker threads
    std::pair<std::vector<float>, float> evaluate(
        const GameState& state,
        std::uint8_t perspective_player,
        std::size_t num_legal_actions
    );
    
    // Batch evaluate multiple states at once (for intra-game parallelism)
    // Returns results in same order as input
    std::vector<std::pair<std::vector<float>, float>> evaluate_batch(
        const std::vector<const GameState*>& states,
        const std::vector<std::uint8_t>& perspective_players,
        const std::vector<std::size_t>& num_legal_actions
    );
    
    // Start the batch processing thread
    void start();
    
    // Stop the batch processing thread
    void stop();
    
    // Get statistics
    std::size_t get_total_requests() const { return total_requests_.load(); }
    std::size_t get_total_batches() const { return total_batches_.load(); }
    float get_average_batch_size() const {
        std::size_t batches = total_batches_.load();
        return batches > 0 ? static_cast<float>(total_requests_.load()) / batches : 0.0f;
    }
    
private:
    // Batch processing thread function
    void process_batches();
    
    // Collect a batch of requests (blocks until batch ready or timeout)
    std::vector<EvaluationRequest> collect_batch();
    
    // Process a batch and send responses
    void process_batch(const std::vector<EvaluationRequest>& batch);
    
    // Configuration
    BatchedEvaluatorConfig config_;
    
    // Neural network callback
    BatchEvaluatorCallback callback_;
    
    // State encoder (thread-safe, stateless)
    StateEncoder encoder_;
    
    // Request queue (protected by mutex)
    std::queue<EvaluationRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Response map (protected by mutex)
    // Maps request_id -> response
    std::unordered_map<std::uint32_t, EvaluationResponse> response_map_;
    std::mutex response_mutex_;
    std::condition_variable response_cv_;
    
    // Request ID counter
    std::atomic<std::uint32_t> next_request_id_{0};
    
    // Batch processing thread
    std::thread batch_thread_;
    std::atomic<bool> running_{false};
    
    // Statistics
    std::atomic<std::size_t> total_requests_{0};
    std::atomic<std::size_t> total_batches_{0};
};

// Helper: Create NNEvaluator compatible with AlphaZero MCTS from BatchedEvaluator
// This allows MCTS to use batched evaluation transparently
inline std::function<alphazero::NNEvaluation(const GameState&)> 
make_batched_nn_evaluator(std::shared_ptr<BatchedEvaluator> evaluator) {
    return [evaluator](const GameState& state) -> alphazero::NNEvaluation {
        // Get legal actions count
        std::vector<Action> legal_actions;
        generate_legal_actions(state, legal_actions);
        
        // Call batched evaluator
        auto [policy, value] = evaluator->evaluate(
            state,
            state.current_player,
            legal_actions.size()
        );
        
        // Return as NNEvaluation
        alphazero::NNEvaluation eval;
        eval.policy = std::move(policy);
        eval.value = value;
        return eval;
    };
}

} // namespace catan
