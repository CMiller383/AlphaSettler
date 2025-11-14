// test_move_gen.cpp
// Test harness for legal move generation.
// Validates rule checking and action enumeration.

#include <iostream>
#include <iomanip>
#include "move_gen.h"
#include "game_state.h"

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

static void print_actions(const std::vector<Action>& actions, std::size_t max_display = 10) {
    std::cout << "  Found " << actions.size() << " legal actions:\n";
    
    std::size_t display_count = std::min(actions.size(), max_display);
    for (std::size_t i = 0; i < display_count; ++i) {
        const Action& a = actions[i];
        std::cout << "    " << std::setw(3) << i << ": " 
                  << action_type_name(a.type);
        
        if (a.location != 0 || a.param1 != 0 || a.param2 != 0) {
            std::cout << " (loc=" << static_cast<int>(a.location)
                      << ", p1=" << static_cast<int>(a.param1)
                      << ", p2=" << static_cast<int>(a.param2) << ")";
        }
        std::cout << "\n";
    }
    
    if (actions.size() > max_display) {
        std::cout << "    ... and " << (actions.size() - max_display) << " more\n";
    }
}

static void test_distance_rule() {
    std::cout << "\n=== Test: Distance Rule ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    bool ok = true;

    // Initially, all vertices should satisfy distance rule (board is empty)
    std::size_t valid_count = 0;
    for (std::uint8_t v = 0; v < NUM_VERTICES; ++v) {
        if (check_distance_rule(game, v)) {
            valid_count++;
        }
    }
    
    std::cout << "  Empty board: " << valid_count << "/" << NUM_VERTICES 
              << " vertices satisfy distance rule\n";
    
    if (valid_count != NUM_VERTICES) {
        std::cout << "  ERROR: All vertices should be valid on empty board\n";
        ok = false;
    }

    // Place a settlement and check adjacent vertices are blocked
    game.place_settlement(0, 10);
    
    std::cout << "  After placing settlement at vertex 10:\n";
    
    // Vertex 10 itself should be blocked
    if (check_distance_rule(game, 10)) {
        std::cout << "    ERROR: Vertex 10 should be blocked (occupied)\n";
        ok = false;
    }
    
    // Find adjacent vertices and verify they're blocked
    const Vertex& vertex = game.board.vertices[10];
    std::size_t adjacent_blocked = 0;
    for (std::uint8_t edge_idx : vertex.edges) {
        if (edge_idx == INVALID_EDGE) continue;
        
        const Edge& edge = game.board.edges[edge_idx];
        for (std::uint8_t adj_v : edge.vertices) {
            if (adj_v == 10 || adj_v == INVALID_VERTEX) continue;
            
            if (!check_distance_rule(game, adj_v)) {
                adjacent_blocked++;
            }
        }
    }
    
    std::cout << "    " << adjacent_blocked << " adjacent vertices blocked\n";
    if (adjacent_blocked == 0) {
        std::cout << "    ERROR: Adjacent vertices should be blocked\n";
        ok = false;
    }

    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_initial_setup_moves() {
    std::cout << "\n=== Test: Initial Setup Move Generation ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    bool ok = true;

    // First action: place initial settlement
    std::vector<Action> actions;
    generate_legal_actions(game, actions);
    
    print_actions(actions, 5);
    
    if (actions.empty()) {
        std::cout << "  ERROR: Should have legal settlement placements\n";
        ok = false;
    }
    
    // All actions should be PlaceInitialSettlement
    for (const Action& a : actions) {
        if (a.type != ActionType::PlaceInitialSettlement) {
            std::cout << "  ERROR: Expected only PlaceInitialSettlement actions\n";
            ok = false;
            break;
        }
    }

    // Place a settlement
    if (!actions.empty()) {
        game.place_settlement(game.current_player, actions[0].location);
        game.setup_settlements_placed[game.current_player]++;
        
        std::cout << "  Placed initial settlement at vertex " 
                  << static_cast<int>(actions[0].location) << "\n";
    }

    // Next action: place initial road
    actions.clear();
    generate_legal_actions(game, actions);
    
    print_actions(actions, 5);
    
    if (actions.empty()) {
        std::cout << "  ERROR: Should have legal road placements\n";
        ok = false;
    }
    
    // All actions should be PlaceInitialRoad
    for (const Action& a : actions) {
        if (a.type != ActionType::PlaceInitialRoad) {
            std::cout << "  ERROR: Expected only PlaceInitialRoad actions\n";
            ok = false;
            break;
        }
    }

    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_road_network_connectivity() {
    std::cout << "\n=== Test: Road Network Connectivity ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    bool ok = true;

    // Place a settlement for player 0 at vertex 10
    game.place_settlement(0, 10);
    
    // Vertex 10 should be connected to player 0's network
    if (!is_connected_to_road_network(game, 0, 10)) {
        std::cout << "  ERROR: Vertex with player's settlement should be connected\n";
        ok = false;
    }
    
    // Other vertices should not be connected yet
    if (is_connected_to_road_network(game, 0, 5)) {
        std::cout << "  ERROR: Unrelated vertex should not be connected\n";
        ok = false;
    }

    // Place a road adjacent to the settlement
    const Vertex& vertex = game.board.vertices[10];
    std::uint8_t road_edge = INVALID_EDGE;
    for (std::uint8_t e : vertex.edges) {
        if (e != INVALID_EDGE) {
            road_edge = e;
            break;
        }
    }
    
    if (road_edge != INVALID_EDGE) {
        game.place_road(0, road_edge);
        std::cout << "  Placed road at edge " << static_cast<int>(road_edge) << "\n";
        
        // Find the other endpoint of this road
        const Edge& edge = game.board.edges[road_edge];
        std::uint8_t other_vertex = INVALID_VERTEX;
        for (std::uint8_t v : edge.vertices) {
            if (v != 10 && v != INVALID_VERTEX) {
                other_vertex = v;
                break;
            }
        }
        
        if (other_vertex != INVALID_VERTEX) {
            // Other endpoint should now be connected
            if (!is_connected_to_road_network(game, 0, other_vertex)) {
                std::cout << "  ERROR: Vertex at end of player's road should be connected\n";
                ok = false;
            } else {
                std::cout << "  Vertex " << static_cast<int>(other_vertex) 
                          << " is now connected via road\n";
            }
        }
    }

    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_main_game_moves() {
    std::cout << "\n=== Test: Main Game Move Generation ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    // Set up game state for main game
    game.game_phase = GamePhase::MainGame;
    game.turn_phase = TurnPhase::Trading;
    
    // Give player 0 some resources
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Brick)] = 5;
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Lumber)] = 5;
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Wool)] = 3;
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Grain)] = 3;
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Ore)] = 3;
    
    // Place an initial settlement and road to enable building
    game.place_settlement(0, 10);
    game.place_road(0, 3);
    
    std::vector<Action> actions;
    generate_legal_actions(game, actions);
    
    print_actions(actions, 15);
    
    bool ok = true;
    
    if (actions.empty()) {
        std::cout << "  ERROR: Should have legal actions in main game\n";
        ok = false;
    }
    
    // Count action types
    std::size_t settlement_count = 0;
    std::size_t road_count = 0;
    std::size_t city_count = 0;
    std::size_t trade_count = 0;
    std::size_t end_turn_count = 0;
    
    for (const Action& a : actions) {
        switch (a.type) {
            case ActionType::PlaceSettlement: settlement_count++; break;
            case ActionType::PlaceRoad:       road_count++; break;
            case ActionType::UpgradeToCity:   city_count++; break;
            case ActionType::BankTrade:       trade_count++; break;
            case ActionType::EndTurn:         end_turn_count++; break;
            default: break;
        }
    }
    
    std::cout << "\n  Action breakdown:\n";
    std::cout << "    Settlements: " << settlement_count << "\n";
    std::cout << "    Roads:       " << road_count << "\n";
    std::cout << "    Cities:      " << city_count << "\n";
    std::cout << "    Trades:      " << trade_count << "\n";
    std::cout << "    End Turn:    " << end_turn_count << "\n";
    
    // Should have at least some roads and trades (player has resources)
    if (road_count == 0) {
        std::cout << "  ERROR: Should have legal road placements\n";
        ok = false;
    }
    
    if (trade_count == 0) {
        std::cout << "  ERROR: Should have legal bank trades (player has 5+ resources)\n";
        ok = false;
    }
    
    // Should have exactly one city upgrade (the one settlement placed)
    if (city_count != 1) {
        std::cout << "  ERROR: Should have exactly 1 city upgrade option, got " 
                  << city_count << "\n";
        ok = false;
    }
    
    // Should have exactly one end turn action
    if (end_turn_count != 1) {
        std::cout << "  ERROR: Should have exactly 1 end turn option, got " 
                  << end_turn_count << "\n";
        ok = false;
    }

    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_bank_trades() {
    std::cout << "\n=== Test: Bank Trade Generation ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    // Give player 0 exactly 4 brick and 8 lumber
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Brick)] = 4;
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Lumber)] = 8;
    
    std::vector<Action> actions;
    generate_bank_trades(game, 0, actions);
    
    print_actions(actions);
    
    bool ok = true;
    
    // Should be able to trade brick for 4 other resources (4 trades)
    // Should be able to trade lumber for 4 other resources (4 trades)
    // Total: 8 trades
    std::size_t expected = 8;
    
    if (actions.size() != expected) {
        std::cout << "  ERROR: Expected " << expected << " trades, got " 
                  << actions.size() << "\n";
        ok = false;
    }
    
    // All should be bank trades
    for (const Action& a : actions) {
        if (a.type != ActionType::BankTrade) {
            std::cout << "  ERROR: Expected only BankTrade actions\n";
            ok = false;
            break;
        }
    }

    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_robber_moves() {
    std::cout << "\n=== Test: Robber Move Generation ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    game.robber_tile = 5; // Place robber on tile 5
    
    std::vector<Action> actions;
    generate_robber_moves(game, actions);
    
    print_actions(actions, 10);
    
    bool ok = true;
    
    // Should be able to move to any tile except tile 5
    std::size_t expected = NUM_TILES - 1;
    
    if (actions.size() != expected) {
        std::cout << "  ERROR: Expected " << expected << " moves, got " 
                  << actions.size() << "\n";
        ok = false;
    }
    
    // None should move to tile 5
    for (const Action& a : actions) {
        if (a.type != ActionType::MoveRobber) {
            std::cout << "  ERROR: Expected only MoveRobber actions\n";
            ok = false;
            break;
        }
        if (a.location == 5) {
            std::cout << "  ERROR: Should not be able to move robber to current tile\n";
            ok = false;
            break;
        }
    }

    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Move Generation Test Suite\n";
    std::cout << "========================================\n";
    
    test_distance_rule();
    test_initial_setup_moves();
    test_road_network_connectivity();
    test_main_game_moves();
    test_bank_trades();
    test_robber_moves();
    
    std::cout << "\n========================================\n";
    std::cout << "  Test Suite Complete!\n";
    std::cout << "========================================\n";
    
    return 0;
}
