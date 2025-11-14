// test_state_transition.cpp
// Test harness for state transitions - validates that actions correctly mutate game state.

#include <iostream>
#include <iomanip>
#include "state_transition.h"
#include "move_gen.h"

using namespace catan;

static const char* action_type_name(ActionType type) {
    switch (type) {
        case ActionType::PlaceInitialSettlement: return "PlaceInitialSettlement";
        case ActionType::PlaceInitialRoad:       return "PlaceInitialRoad";
        case ActionType::RollDice:               return "RollDice";
        case ActionType::PlaceSettlement:        return "PlaceSettlement";
        case ActionType::PlaceRoad:              return "PlaceRoad";
        case ActionType::UpgradeToCity:          return "UpgradeToCity";
        case ActionType::BuyDevCard:             return "BuyDevCard";
        case ActionType::BankTrade:              return "BankTrade";
        case ActionType::MoveRobber:             return "MoveRobber";
        case ActionType::StealFromPlayer:        return "StealFromPlayer";
        case ActionType::DiscardResources:       return "DiscardResources";
        case ActionType::EndTurn:                return "EndTurn";
        default:                                 return "???";
    }
}

static void test_setup_phase() {
    std::cout << "\n=== Test: Setup Phase Transitions ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    bool ok = true;
    
    // Player 0 should start
    if (game.current_player != 0) {
        std::cout << "  ERROR: Expected player 0 to start\n";
        ok = false;
    }
    
    // Get legal actions (should be initial settlement placements)
    std::vector<Action> actions;
    generate_legal_actions(game, actions);
    
    if (actions.empty() || actions[0].type != ActionType::PlaceInitialSettlement) {
        std::cout << "  ERROR: Expected initial settlement placements\n";
        ok = false;
    }
    
    // Place initial settlement
    apply_action(game, actions[0], 111);
    
    std::cout << "  Player 0 placed initial settlement at vertex " 
              << static_cast<int>(actions[0].location) << "\n";
    
    if (game.setup_settlements_placed[0] != 1) {
        std::cout << "  ERROR: Setup settlement count should be 1\n";
        ok = false;
    }
    
    if (!game.is_vertex_occupied(actions[0].location)) {
        std::cout << "  ERROR: Vertex should be occupied\n";
        ok = false;
    }
    
    // Next action should be road placement
    actions.clear();
    generate_legal_actions(game, actions);
    
    if (actions.empty() || actions[0].type != ActionType::PlaceInitialRoad) {
        std::cout << "  ERROR: Expected initial road placements\n";
        ok = false;
    }
    
    // Place initial road
    apply_action(game, actions[0], 222);
    
    std::cout << "  Player 0 placed initial road at edge " 
              << static_cast<int>(actions[0].location) << "\n";
    
    if (game.setup_roads_placed[0] != 1) {
        std::cout << "  ERROR: Setup road count should be 1\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_resource_costs() {
    std::cout << "\n=== Test: Resource Costs ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    game.game_phase = GamePhase::MainGame;
    game.turn_phase = TurnPhase::Trading;
    
    // Give player 0 exact resources for a road
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Brick)] = 1;
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Lumber)] = 1;
    
    // Place a settlement first to have road network
    game.place_settlement(0, 10);
    
    bool ok = true;
    
    // Try to place a road
    Action road_action = Action::place_road(3);
    apply_action(game, road_action, 333);
    
    // Resources should be consumed
    if (game.players[0].resources[static_cast<std::size_t>(ResourceType::Brick)] != 0 ||
        game.players[0].resources[static_cast<std::size_t>(ResourceType::Lumber)] != 0) {
        std::cout << "  ERROR: Resources should be consumed after building\n";
        ok = false;
    }
    
    // Road should be placed
    if (!game.is_edge_occupied(3)) {
        std::cout << "  ERROR: Road should be placed\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_trading() {
    std::cout << "\n=== Test: Trading ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    // Give player 0 exactly 4 brick
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Brick)] = 4;
    
    bool ok = true;
    
    std::uint8_t initial_brick = game.players[0].resources[static_cast<std::size_t>(ResourceType::Brick)];
    std::uint8_t initial_lumber = game.players[0].resources[static_cast<std::size_t>(ResourceType::Lumber)];
    
    // Trade brick for lumber (4:1)
    Action trade = Action::bank_trade(ResourceType::Brick, ResourceType::Lumber);
    apply_action(game, trade, 444);
    
    std::uint8_t final_brick = game.players[0].resources[static_cast<std::size_t>(ResourceType::Brick)];
    std::uint8_t final_lumber = game.players[0].resources[static_cast<std::size_t>(ResourceType::Lumber)];
    
    std::cout << "  Before trade: " << static_cast<int>(initial_brick) << " brick, " 
              << static_cast<int>(initial_lumber) << " lumber\n";
    std::cout << "  After trade:  " << static_cast<int>(final_brick) << " brick, " 
              << static_cast<int>(final_lumber) << " lumber\n";
    
    if (final_brick != 0 || final_lumber != 1) {
        std::cout << "  ERROR: Trade should consume 4 brick and give 1 lumber\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_dice_and_resources() {
    std::cout << "\n=== Test: Dice Roll and Resource Distribution ===\n";
    
    GameState game = GameState::create_new_game(4, 54321);
    game.game_phase = GamePhase::MainGame;
    game.turn_phase = TurnPhase::RollDice;
    
    // Place settlements for players
    game.place_settlement(0, 0);
    game.place_settlement(1, 10);
    
    bool ok = true;
    
    std::uint8_t initial_resources[2] = {
        game.players[0].total_resources(),
        game.players[1].total_resources()
    };
    
    // Roll dice
    Action roll = Action::roll_dice();
    apply_action(game, roll, 999);
    
    std::cout << "  Rolled: " << static_cast<int>(game.last_dice_roll) << "\n";
    
    std::uint8_t final_resources[2] = {
        game.players[0].total_resources(),
        game.players[1].total_resources()
    };
    
    std::cout << "  Player 0: " << static_cast<int>(initial_resources[0]) 
              << " -> " << static_cast<int>(final_resources[0]) << " resources\n";
    std::cout << "  Player 1: " << static_cast<int>(initial_resources[1])
              << " -> " << static_cast<int>(final_resources[1]) << " resources\n";
    
    // Phase should transition to Trading (unless rolled 7)
    if (game.last_dice_roll != 7 && game.turn_phase != TurnPhase::Trading) {
        std::cout << "  ERROR: Should transition to Trading phase after non-7 roll\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_dev_cards() {
    std::cout << "\n=== Test: Development Cards ===\n";
    
    GameState game = GameState::create_new_game(4, 11111);
    
    // Give player 0 resources to buy a dev card
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Wool)] = 5;
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Grain)] = 5;
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Ore)] = 5;
    
    bool ok = true;
    
    std::uint32_t initial_deck_size = 0;
    for (std::uint8_t count : game.dev_deck) {
        initial_deck_size += count;
    }
    
    std::cout << "  Initial deck size: " << initial_deck_size << "\n";
    
    // Buy a dev card
    Action buy = Action::buy_dev_card();
    apply_action(game, buy, 777);
    
    std::uint32_t final_deck_size = 0;
    for (std::uint8_t count : game.dev_deck) {
        final_deck_size += count;
    }
    
    std::cout << "  Final deck size: " << final_deck_size << "\n";
    
    if (final_deck_size != initial_deck_size - 1) {
        std::cout << "  ERROR: Deck should have one less card\n";
        ok = false;
    }
    
    // Resources should be consumed
    if (game.players[0].resources[static_cast<std::size_t>(ResourceType::Wool)] != 4 ||
        game.players[0].resources[static_cast<std::size_t>(ResourceType::Grain)] != 4 ||
        game.players[0].resources[static_cast<std::size_t>(ResourceType::Ore)] != 4) {
        std::cout << "  ERROR: Resources should be consumed\n";
        ok = false;
    }
    
    // Card should be in "bought this turn" hand
    std::uint8_t cards_bought = 0;
    for (std::uint8_t count : game.players[0].dev_cards_bought_this_turn) {
        cards_bought += count;
    }
    
    if (cards_bought != 1) {
        std::cout << "  ERROR: Should have 1 card in 'bought this turn' hand\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_victory_points() {
    std::cout << "\n=== Test: Victory Points ===\n";
    
    GameState game = GameState::create_new_game(4, 88888);
    game.game_phase = GamePhase::MainGame;
    
    bool ok = true;
    
    // Place settlements and cities for player 0
    game.place_settlement(0, 5);
    game.update_victory_points(0);
    
    if (game.players[0].public_victory_points != 1) {
        std::cout << "  ERROR: 1 settlement should give 1 VP, got " 
                  << static_cast<int>(game.players[0].public_victory_points) << "\n";
        ok = false;
    }
    
    std::cout << "  After 1 settlement: " << static_cast<int>(game.players[0].public_victory_points) << " VP\n";
    
    // Place another settlement
    game.place_settlement(0, 15);
    game.update_victory_points(0);
    
    if (game.players[0].public_victory_points != 2) {
        std::cout << "  ERROR: 2 settlements should give 2 VP, got "
                  << static_cast<int>(game.players[0].public_victory_points) << "\n";
        ok = false;
    }
    
    std::cout << "  After 2 settlements: " << static_cast<int>(game.players[0].public_victory_points) << " VP\n";
    
    // Upgrade one to city
    game.upgrade_to_city(0, 5);
    game.update_victory_points(0);
    
    if (game.players[0].public_victory_points != 3) {
        std::cout << "  ERROR: 1 city + 1 settlement should give 3 VP, got "
                  << static_cast<int>(game.players[0].public_victory_points) << "\n";
        ok = false;
    }
    
    std::cout << "  After upgrading to city: " << static_cast<int>(game.players[0].public_victory_points) << " VP\n";
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_turn_management() {
    std::cout << "\n=== Test: Turn Management ===\n";
    
    GameState game = GameState::create_new_game(4, 99999);
    game.game_phase = GamePhase::MainGame;
    game.turn_phase = TurnPhase::Trading;
    
    bool ok = true;
    
    std::uint8_t initial_player = game.current_player;
    std::uint32_t initial_turn = game.turn_number;
    
    std::cout << "  Initial: Player " << static_cast<int>(initial_player) 
              << ", Turn " << initial_turn << "\n";
    
    // End turn
    Action end = Action::end_turn();
    apply_action(game, end, 0);
    
    std::uint8_t final_player = game.current_player;
    std::uint32_t final_turn = game.turn_number;
    
    std::cout << "  After end_turn: Player " << static_cast<int>(final_player)
              << ", Turn " << final_turn << "\n";
    
    if (final_player != (initial_player + 1) % 4) {
        std::cout << "  ERROR: Should advance to next player\n";
        ok = false;
    }
    
    // Turn number should only increment when wrapping to player 0
    if (initial_player == 3 && final_turn != initial_turn + 1) {
        std::cout << "  ERROR: Turn number should increment when wrapping to player 0\n";
        ok = false;
    }
    
    if (game.turn_phase != TurnPhase::RollDice) {
        std::cout << "  ERROR: Turn phase should reset to RollDice\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_full_game_simulation() {
    std::cout << "\n=== Test: Full Game Simulation ===\n";
    
    GameState game = GameState::create_new_game(4, 42);
    
    bool ok = true;
    int action_count = 0;
    const int max_actions = 50; // Limit iterations for testing
    
    std::cout << "  Simulating game with up to " << max_actions << " actions...\n";
    
    while (action_count < max_actions && !game.is_game_over()) {
        std::vector<Action> actions;
        generate_legal_actions(game, actions);
        
        if (actions.empty()) {
            std::cout << "  ERROR: No legal actions available but game not over\n";
            std::cout << "    Current player: " << static_cast<int>(game.current_player) << "\n";
            std::cout << "    Game phase: " << (game.game_phase == GamePhase::Setup ? "Setup" :
                                                 game.game_phase == GamePhase::MainGame ? "MainGame" : "Finished") << "\n";
            std::cout << "    Setup progress:\n";
            for (std::uint8_t p = 0; p < game.num_players; ++p) {
                std::cout << "      Player " << static_cast<int>(p) 
                          << ": settlements=" << static_cast<int>(game.setup_settlements_placed[p])
                          << ", roads=" << static_cast<int>(game.setup_roads_placed[p]) << "\n";
            }
            ok = false;
            break;
        }
        
        // Debug output for setup phase
        if (game.game_phase == GamePhase::Setup) {
            std::cout << "  Action " << action_count 
                      << ": Player " << static_cast<int>(game.current_player)
                      << " - " << action_type_name(actions[0].type) << "\n";
        }
        
        // Pick first legal action (not random, but deterministic for testing)
        apply_action(game, actions[0], action_count * 123);
        action_count++;
    }
    
    std::cout << "  Executed " << action_count << " actions\n";
    std::cout << "  Game phase: " << (game.game_phase == GamePhase::Setup ? "Setup" :
                                       game.game_phase == GamePhase::MainGame ? "MainGame" : "Finished") << "\n";
    
    // Show final setup state
    if (game.game_phase == GamePhase::MainGame || game.game_phase == GamePhase::Finished) {
        std::cout << "  Final setup state:\n";
        for (std::uint8_t p = 0; p < game.num_players; ++p) {
            std::cout << "    Player " << static_cast<int>(p) 
                      << ": settlements=" << static_cast<int>(game.setup_settlements_placed[p])
                      << ", roads=" << static_cast<int>(game.setup_roads_placed[p])
                      << ", VP=" << static_cast<int>(game.players[p].public_victory_points) << "\n";
        }
    }
    
    if (game.is_game_over()) {
        std::cout << "  Winner: Player " << static_cast<int>(game.get_winner()) << "\n";
    }
    
    if (action_count == 0) {
        std::cout << "  ERROR: No actions were executed\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  State Transition Test Suite\n";
    std::cout << "========================================\n";
    
    test_setup_phase();
    test_resource_costs();
    test_trading();
    test_dice_and_resources();
    test_dev_cards();
    test_victory_points();
    test_turn_management();
    test_full_game_simulation();
    
    std::cout << "\n========================================\n";
    std::cout << "  Test Suite Complete!\n";
    std::cout << "========================================\n";
    
    return 0;
}
