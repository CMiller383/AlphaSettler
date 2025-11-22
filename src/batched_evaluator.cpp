// batched_evaluator.cpp
// Implementation of thread-safe batched neural network evaluator.

#include "batched_evaluator.h"
#include <chrono>
#include <algorithm>
#include <string>
#include <stdexcept>

namespace catan {

BatchedEvaluator::BatchedEvaluator(
    const BatchedEvaluatorConfig& config,
    BatchEvaluatorCallback callback
)
    : config_(config)
    , callback_(std::move(callback))
    , encoder_()
{}

BatchedEvaluator::~BatchedEvaluator() {
    stop();
}

void BatchedEvaluator::start() {
    if (running_.load()) {
        return; // Already running
    }
    
    running_.store(true);
    batch_thread_ = std::thread(&BatchedEvaluator::process_batches, this);
}

void BatchedEvaluator::stop() {
    if (!running_.load()) {
        return; // Not running
    }
    
    running_.store(false);
    queue_cv_.notify_all();
    
    if (batch_thread_.joinable()) {
        batch_thread_.join();
    }
}

std::pair<std::vector<float>, float> BatchedEvaluator::evaluate(
    const GameState& state,
    std::uint8_t perspective_player,
    std::size_t num_legal_actions
) {
    // If batching disabled, evaluate immediately
    if (!config_.enable_batching) {
        std::vector<float> encoded = encoder_.encode_state(state, perspective_player);
        std::size_t feature_size = encoder_.get_feature_size();
        std::vector<std::size_t> num_actions_vec = {num_legal_actions};
        auto batch_result = callback_(encoded, num_actions_vec, 1, feature_size);
        return batch_result[0];
    }
    
    // Generate unique request ID
    std::uint32_t request_id = next_request_id_.fetch_add(1);
    
    // Submit request to queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push({state, perspective_player, request_id});
        total_requests_.fetch_add(1);
    }
    queue_cv_.notify_one();
    
    // Wait for response with timeout
    std::unique_lock<std::mutex> response_lock(response_mutex_);
    auto timeout = std::chrono::milliseconds(30000);  // 30 second timeout (generous for GPU inference)
    bool got_response = response_cv_.wait_for(response_lock, timeout, [this, request_id]() {
        return response_map_.count(request_id) > 0;
    });
    
    if (!got_response) {
        throw std::runtime_error("Batched evaluator timeout waiting for response. " +
                                 std::string("Request ID: ") + std::to_string(request_id));
    }
    
    // Extract response
    EvaluationResponse response = std::move(response_map_[request_id]);
    response_map_.erase(request_id);
    
    return {std::move(response.policy), response.value};
}

std::vector<std::pair<std::vector<float>, float>> BatchedEvaluator::evaluate_batch(
    const std::vector<const GameState*>& states,
    const std::vector<std::uint8_t>& perspective_players,
    const std::vector<std::size_t>& num_legal_actions
) {
    if (states.size() != perspective_players.size() || states.size() != num_legal_actions.size()) {
        throw std::invalid_argument("evaluate_batch: size mismatch between inputs");
    }
    
    // OPTIMIZATION: Pre-stack encoded states into flat array to avoid Python list conversion
    std::size_t batch_size = states.size();
    std::size_t feature_size = encoder_.get_feature_size();
    
    // Allocate flat array for all states: batch_size * feature_size
    std::vector<float> stacked_states;
    stacked_states.reserve(batch_size * feature_size);
    
    // Encode and stack states directly
    for (std::size_t i = 0; i < batch_size; ++i) {
        std::vector<float> encoded = encoder_.encode_state(*states[i], perspective_players[i]);
        stacked_states.insert(stacked_states.end(), encoded.begin(), encoded.end());
    }
    
    // Extract num_legal_actions into separate vector
    std::vector<std::size_t> num_actions_vec(num_legal_actions.begin(), num_legal_actions.end());
    
    // Call callback with pre-stacked data (caller holds GIL, so this is safe)
    return callback_(stacked_states, num_actions_vec, batch_size, feature_size);
}

void BatchedEvaluator::process_batches() {
    while (running_.load()) {
        // Collect batch of requests
        std::vector<EvaluationRequest> batch = collect_batch();
        
        if (batch.empty()) {
            continue; // Timeout or shutdown
        }
        
        // Process batch
        process_batch(batch);
        total_batches_.fetch_add(1);
    }
}

std::vector<EvaluationRequest> BatchedEvaluator::collect_batch() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Wait for requests with short timeout for responsiveness
    auto timeout = std::chrono::milliseconds(config_.timeout_ms);
    queue_cv_.wait_for(lock, timeout, [this]() {
        return !request_queue_.empty() || !running_.load();
    });
    
    if (!running_.load() && request_queue_.empty()) {
        return {};
    }
    
    // If no requests, return empty
    if (request_queue_.empty()) {
        return {};
    }
    
    // Collect up to max_batch_size requests
    std::vector<EvaluationRequest> batch;
    batch.reserve(std::min(config_.max_batch_size, request_queue_.size()));
    
    while (!request_queue_.empty() && batch.size() < config_.max_batch_size) {
        batch.push_back(std::move(request_queue_.front()));
        request_queue_.pop();
    }
    
    return batch;
}

void BatchedEvaluator::process_batch(const std::vector<EvaluationRequest>& batch) {
    if (batch.empty()) {
        return;
    }
    
    // OPTIMIZATION: Pre-stack encoded states into flat array
    std::size_t batch_size = batch.size();
    std::size_t feature_size = encoder_.get_feature_size();
    
    std::vector<float> stacked_states;
    stacked_states.reserve(batch_size * feature_size);
    
    std::vector<std::size_t> num_legal_actions_vec;
    num_legal_actions_vec.reserve(batch_size);
    
    for (const auto& request : batch) {
        // Encode state
        std::vector<float> encoded = encoder_.encode_state(
            request.state,
            request.perspective_player
        );
        
        // Stack into flat array
        stacked_states.insert(stacked_states.end(), encoded.begin(), encoded.end());
        
        // Get number of legal actions for this state
        std::vector<Action> legal_actions;
        generate_legal_actions(request.state, legal_actions);
        num_legal_actions_vec.push_back(legal_actions.size());
    }
    
    // Call neural network with pre-stacked states
    std::vector<std::pair<std::vector<float>, float>> results = 
        callback_(stacked_states, num_legal_actions_vec, batch_size, feature_size);
    
    // Distribute responses
    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        
        for (std::size_t i = 0; i < batch_size && i < results.size(); ++i) {
            EvaluationResponse response;
            response.request_id = batch[i].request_id;
            response.policy = std::move(results[i].first);
            response.value = results[i].second;
            
            response_map_[response.request_id] = std::move(response);
        }
    }
    
    response_cv_.notify_all();
}

} // namespace catan
