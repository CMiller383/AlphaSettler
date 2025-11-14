// test_mcts.cpp
// Test harness for pure MCTS implementation.

#include <iostream>
#include <iomanip>
#include "mcts/mcts_search.h"
#include "game_state.h"
#include "move_gen.h"
#include "state_transition.h"

using namespace catan;
using namespace catan::mcts;

static void test_mcts_basic() {
    std::cout << "\n=== Test: Basic MCTS Search ===\n";
    
    // Create a game in setup phase
    GameState game = GameState::create_new_game(4, 12345);
    
    // Configure MCTS with fewer iterations for testing
    MCTSConfig config;
    config.num_iterations = 100;
    config.exploration_constant = 1.41;
    config.random_seed = 42;
    
    MCTSSearch mcts(config);
    
    bool ok = true;
    
    // Run MCTS search
    Action action = mcts.search(game);
    
    std::cout << "  MCTS selected action type: " << static_cast<int>(action.type) << "\n";
    
    // Verify action is legal
    std::vector<Action> legal_actions;
    generate_legal_actions(game, legal_actions);
    
    bool found = false;
    for (const Action& legal_action : legal_actions) {
        if (legal_action.type == action.type && legal_action.location == action.location) {
            found = true;
            break;
        }
    }
    
    if (!found) {
        std::cout << "  ERROR: MCTS selected illegal action\n";
        ok = false;
    }
    
    // Check root statistics
    const MCTSNode* root = mcts.get_root();
    std::cout << "  Root visits: " << root->visit_count << "\n";
    std::cout << "  Root children: " << root->children.size() << "\n";
    
    if (root->visit_count != config.num_iterations) {
        std::cout << "  ERROR: Root visits should equal num_iterations\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_mcts_self_play() {
    std::cout << "\n=== Test: MCTS Self-Play Game ===\n";
    
    GameState game = GameState::create_new_game(4, 54321);
    
    MCTSConfig config;
    config.num_iterations = 50;  // Fast iterations for testing
    config.exploration_constant = 1.41;
    config.random_seed = 99;
    
    MCTSSearch mcts(config);
    
    bool ok = true;
    int action_count = 0;
    const int max_actions = 100; // Limit to prevent infinite loops
    
    std::cout << "  Playing game with MCTS agent...\n";
    
    while (!game.is_game_over() && action_count < max_actions) {
        // Get legal actions
        std::vector<Action> legal_actions;
        generate_legal_actions(game, legal_actions);
        
        if (legal_actions.empty()) {
            std::cout << "  ERROR: No legal actions but game not over\n";
            ok = false;
            break;
        }
        
        // Run MCTS to select action
        Action action = mcts.search(game);
        
        // Show first few moves
        if (action_count < 5) {
            std::cout << "    Move " << action_count 
                      << ": Player " << static_cast<int>(game.current_player)
                      << " action type " << static_cast<int>(action.type) << "\n";
        }
        
        // Apply action
        apply_action(game, action, action_count * 123);
        action_count++;
    }
    
    std::cout << "  Completed " << action_count << " actions\n";
    std::cout << "  Game phase: " << (game.game_phase == GamePhase::Setup ? "Setup" :
                                       game.game_phase == GamePhase::MainGame ? "MainGame" : "Finished") << "\n";
    
    // Show final state
    if (game.is_game_over()) {
        std::cout << "  Winner: Player " << static_cast<int>(game.get_winner()) << "\n";
    } else {
        std::cout << "  Game incomplete (reached max actions)\n";
    }
    
    // Show victory points
    std::cout << "  Final scores:\n";
    for (std::uint8_t p = 0; p < game.num_players; ++p) {
        std::cout << "    Player " << static_cast<int>(p) 
                  << ": " << static_cast<int>(game.players[p].total_victory_points()) << " VP\n";
    }
    
    if (action_count == 0) {
        std::cout << "  ERROR: No actions executed\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_mcts_determinism() {
    std::cout << "\n=== Test: MCTS Determinism ===\n";
    
    GameState game = GameState::create_new_game(4, 11111);
    
    MCTSConfig config;
    config.num_iterations = 50;
    config.random_seed = 777; // Fixed seed
    
    bool ok = true;
    
    // Run MCTS twice with same seed
    MCTSSearch mcts1(config);
    Action action1 = mcts1.search(game);
    
    MCTSSearch mcts2(config);
    Action action2 = mcts2.search(game);
    
    std::cout << "  First run: action type " << static_cast<int>(action1.type) 
              << ", location " << static_cast<int>(action1.location) << "\n";
    std::cout << "  Second run: action type " << static_cast<int>(action2.type)
              << ", location " << static_cast<int>(action2.location) << "\n";
    
    if (action1.type != action2.type || action1.location != action2.location) {
        std::cout << "  ERROR: MCTS should be deterministic with fixed seed\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Pure MCTS Test Suite\n";
    std::cout << "========================================\n";
    
    test_mcts_basic();
    test_mcts_self_play();
    test_mcts_determinism();
    
    std::cout << "\n========================================\n";
    std::cout << "  MCTS Test Suite Complete!\n";
    std::cout << "========================================\n";
    
    return 0;
}
