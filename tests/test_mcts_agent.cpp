// test_mcts_agent.cpp
// Test MCTS agent interface and self-play capabilities.

#include <iostream>
#include <iomanip>
#include <chrono>
#include "mcts/mcts_agent.h"

using namespace catan;
using namespace catan::mcts;

static void test_agent_basic() {
    std::cout << "\n=== Test: MCTS Agent Basic ===\n";
    
    MCTSConfig config;
    config.num_iterations = 100;
    config.random_seed = 42;
    
    MCTSAgent agent(config);
    GameState game = GameState::create_new_game(4, 12345);
    
    bool ok = true;
    
    // Select action
    Action action = agent.select_action(game);
    std::cout << "  Selected action type: " << static_cast<int>(action.type) << "\n";
    
    // Get policy
    std::vector<ActionPolicy> policy = agent.get_action_policy(game);
    std::cout << "  Policy size: " << policy.size() << "\n";
    
    // Verify probabilities sum to ~1.0
    double prob_sum = 0.0;
    for (const auto& ap : policy) {
        prob_sum += ap.probability;
    }
    
    std::cout << "  Probability sum: " << prob_sum << "\n";
    
    if (prob_sum < 0.99 || prob_sum > 1.01) {
        std::cout << "  ERROR: Probabilities should sum to 1.0\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_agent_full_game() {
    std::cout << "\n=== Test: Agent Full Game ===\n";
    
    MCTSConfig config;
    config.num_iterations = 50;  // Fast for testing
    config.random_seed = 99;
    
    MCTSAgent agent(config);
    GameState game = GameState::create_new_game(4, 54321);
    
    bool ok = true;
    
    std::uint8_t winner = agent.play_game(game, 5000);
    
    std::cout << "  Game finished: " << (winner != 0xFF ? "Yes" : "No") << "\n";
    if (winner != 0xFF) {
        std::cout << "  Winner: Player " << static_cast<int>(winner) << "\n";
    }
    
    std::cout << "  Turn number: " << game.turn_number << "\n";
    
    std::cout << "  Final scores:\n";
    for (std::uint8_t p = 0; p < game.num_players; ++p) {
        std::cout << "    Player " << static_cast<int>(p) 
                  << ": " << static_cast<int>(game.players[p].total_victory_points()) << " VP\n";
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_self_play_engine() {
    std::cout << "\n=== Test: Self-Play Engine ===\n";
    
    MCTSConfig config;
    config.num_iterations = 30;  // Fast for testing
    config.random_seed = 777;
    
    SelfPlayEngine engine(config);
    
    const std::uint32_t num_games = 5;
    std::cout << "  Playing " << num_games << " games...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    SelfPlayStats stats = engine.play_games(num_games, 4, 12345);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    bool ok = true;
    
    std::cout << "  Games played: " << stats.games_played << "\n";
    std::cout << "  Games finished: " << stats.games_finished << "\n";
    std::cout << "  Win distribution:\n";
    for (std::uint8_t p = 0; p < 4; ++p) {
        std::cout << "    Player " << static_cast<int>(p) 
                  << ": " << stats.wins_by_player[p] << " wins\n";
    }
    std::cout << "  Avg game length: " << std::fixed << std::setprecision(1) 
              << stats.avg_game_length << " turns\n";
    std::cout << "  Avg final VP: " << std::fixed << std::setprecision(2) 
              << stats.avg_final_vp << "\n";
    std::cout << "  Time taken: " << duration.count() << " ms\n";
    std::cout << "  Time per game: " << (duration.count() / num_games) << " ms\n";
    
    if (stats.games_played != num_games) {
        std::cout << "  ERROR: Should have played " << num_games << " games\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_mcts_performance() {
    std::cout << "\n=== Test: MCTS Performance Benchmark ===\n";
    
    GameState game = GameState::create_new_game(4, 99999);
    
    // Test different iteration counts
    std::vector<std::uint32_t> iteration_counts = {50, 100, 200, 500};
    
    for (std::uint32_t iters : iteration_counts) {
        MCTSConfig config;
        config.num_iterations = iters;
        config.random_seed = 42;
        
        MCTSSearch mcts(config);
        
        auto start = std::chrono::high_resolution_clock::now();
        Action action = mcts.search(game);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        const MCTSNode* root = mcts.get_root();
        
        std::cout << "  " << iters << " iterations:\n";
        std::cout << "    Time: " << duration.count() << " ms\n";
        std::cout << "    Root children: " << root->children.size() << "\n";
        
        // Count total nodes in tree
        std::uint32_t total_nodes = 1; // root
        std::uint32_t max_depth = 0;
        for (const auto& child : root->children) {
            total_nodes++;
            total_nodes += static_cast<std::uint32_t>(child->children.size());
            if (!child->children.empty()) max_depth = std::max(max_depth, 2u);
        }
        
        std::cout << "    Total nodes: " << total_nodes << "\n";
        std::cout << "    Max depth: " << max_depth << "\n";
        std::cout << "    Action selected: type " << static_cast<int>(action.type) << "\n";
        
        if (duration.count() > 0) {
            double iter_per_ms = static_cast<double>(iters) / duration.count();
            std::cout << "    Throughput: " << std::fixed << std::setprecision(1) 
                      << (iter_per_ms * 1000.0) << " iterations/sec\n";
        }
    }
    
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  MCTS Agent Test Suite\n";
    std::cout << "========================================\n";
    
    test_agent_basic();
    test_agent_full_game();
    test_self_play_engine();
    test_mcts_performance();
    
    std::cout << "\n========================================\n";
    std::cout << "  MCTS Agent Test Suite Complete!\n";
    std::cout << "========================================\n";
    
    return 0;
}
