// batched_evaluator.cpp
// Implementation of thread-safe batched neural network evaluator.

#include "batched_evaluator.h"
#include <chrono>
#include <algorithm>
#include <iostream>

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
    
    std::cout << "[DEBUG] BatchedEvaluator::start() - starting thread" << std::endl;
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
    std::cout << "[DEBUG] evaluate() called, enable_batching=" << config_.enable_batching << std::endl;
    
    // If batching disabled, evaluate immediately
    if (!config_.enable_batching) {
        std::cout << "[DEBUG] Batching disabled, direct callback" << std::endl;
        std::vector<float> encoded = encoder_.encode_state(state, perspective_player);
        auto batch_result = callback_({{encoded, num_legal_actions}});
        return batch_result[0];
    }
    
    std::cout << "[DEBUG] Batching enabled, submitting request" << std::endl;
    
    // Generate unique request ID
    std::uint32_t request_id = next_request_id_.fetch_add(1);
    std::cout << "[DEBUG] Request ID: " << request_id << std::endl;
    
    // Submit request to queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push({state, perspective_player, request_id});
        total_requests_.fetch_add(1);
        std::cout << "[DEBUG] Request queued, queue size: " << request_queue_.size() << std::endl;
    }
    queue_cv_.notify_one();
    std::cout << "[DEBUG] Queue notified" << std::endl;
    
    // Wait for response
    std::cout << "[DEBUG] Waiting for response..." << std::endl;
    std::unique_lock<std::mutex> response_lock(response_mutex_);
    response_cv_.wait(response_lock, [this, request_id]() {
        bool ready = response_map_.count(request_id) > 0;
        if (ready) {
            std::cout << "[DEBUG] Response ready for request " << request_id << std::endl;
        }
        return ready;
    });
    
    std::cout << "[DEBUG] Response received for request " << request_id << std::endl;
    
    // Extract response
    EvaluationResponse response = std::move(response_map_[request_id]);
    response_map_.erase(request_id);
    
    return {std::move(response.policy), response.value};
}

void BatchedEvaluator::process_batches() {
    std::cout << "[DEBUG] Batch processing thread started" << std::endl;
    
    while (running_.load()) {
        std::cout << "[DEBUG] Collecting batch..." << std::endl;
        
        // Collect batch of requests
        std::vector<EvaluationRequest> batch = collect_batch();
        
        std::cout << "[DEBUG] Batch collected, size: " << batch.size() << std::endl;
        
        if (batch.empty()) {
            std::cout << "[DEBUG] Empty batch, continuing..." << std::endl;
            continue; // Timeout or shutdown
        }
        
        // Process batch
        std::cout << "[DEBUG] Processing batch..." << std::endl;
        process_batch(batch);
        total_batches_.fetch_add(1);
        std::cout << "[DEBUG] Batch processed" << std::endl;
    }
    
    std::cout << "[DEBUG] Batch processing thread stopped" << std::endl;
}

std::vector<EvaluationRequest> BatchedEvaluator::collect_batch() {
    std::vector<EvaluationRequest> batch;
    batch.reserve(config_.max_batch_size);
    
    std::cout << "[DEBUG] collect_batch() called" << std::endl;
    
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    std::cout << "[DEBUG] Queue lock acquired, queue size: " << request_queue_.size() << std::endl;
    
    // Wait for at least one request or timeout
    auto deadline = std::chrono::steady_clock::now() + 
                    std::chrono::milliseconds(config_.timeout_ms);
    
    // Wait for first request
    while (running_.load() && request_queue_.empty()) {
        std::cout << "[DEBUG] Queue empty, waiting..." << std::endl;
        if (queue_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            std::cout << "[DEBUG] Timeout waiting for requests" << std::endl;
            return batch;  // Empty batch on timeout
        }
        std::cout << "[DEBUG] Woke up, queue size: " << request_queue_.size() << std::endl;
    }
    
    std::cout << "[DEBUG] Collecting requests from queue..." << std::endl;
    
    // Collect requests up to max batch size
    while (!request_queue_.empty() && batch.size() < config_.max_batch_size) {
        batch.push_back(std::move(request_queue_.front()));
        request_queue_.pop();
        
        // If we have min_batch_size, we can process immediately
        if (batch.size() >= config_.min_batch_size) {
            std::cout << "[DEBUG] Min batch size reached: " << batch.size() << std::endl;
            break;
        }
    }
    
    std::cout << "[DEBUG] Collected " << batch.size() << " requests" << std::endl;
    
    return batch;
}

void BatchedEvaluator::process_batch(const std::vector<EvaluationRequest>& batch) {
    if (batch.empty()) {
        return;
    }
    
    std::cout << "[DEBUG] process_batch() called with " << batch.size() << " requests" << std::endl;
    
    // Encode all states in batch
    std::vector<std::pair<std::vector<float>, std::size_t>> encoded_batch;
    encoded_batch.reserve(batch.size());
    
    for (const auto& request : batch) {
        std::vector<float> encoded = encoder_.encode_state(
            request.state,
            request.perspective_player
        );
        
        // Get number of legal actions for this state
        std::vector<Action> legal_actions;
        generate_legal_actions(request.state, legal_actions);
        
        encoded_batch.emplace_back(std::move(encoded), legal_actions.size());
    }
    
    std::cout << "[DEBUG] States encoded, calling NN callback..." << std::endl;
    
    // Call neural network with batched states
    std::vector<std::pair<std::vector<float>, float>> results = callback_(encoded_batch);
    
    std::cout << "[DEBUG] NN callback returned " << results.size() << " results" << std::endl;
    
    // Distribute responses
    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        
        for (std::size_t i = 0; i < batch.size() && i < results.size(); ++i) {
            EvaluationResponse response;
            response.request_id = batch[i].request_id;
            response.policy = std::move(results[i].first);
            response.value = results[i].second;
            
            response_map_[response.request_id] = std::move(response);
            std::cout << "[DEBUG] Response stored for request " << batch[i].request_id << std::endl;
        }
    }
    
    std::cout << "[DEBUG] Notifying waiting threads..." << std::endl;
    response_cv_.notify_all();
    std::cout << "[DEBUG] Threads notified" << std::endl;
}

} // namespace catan
