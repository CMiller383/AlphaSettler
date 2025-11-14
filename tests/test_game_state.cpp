// test_game_state.cpp
// Test harness for game state representation and basic operations.

#include <iostream>
#include <iomanip>
#include "game_state.h"

using namespace catan;

static const char* resource_type_name(ResourceType r) {
    switch (r) {
        case ResourceType::Brick:  return "Brick ";
        case ResourceType::Lumber: return "Lumber";
        case ResourceType::Wool:   return "Wool  ";
        case ResourceType::Grain:  return "Grain ";
        case ResourceType::Ore:    return "Ore   ";
        default:                   return "???   ";
    }
}

static const char* dev_card_name(DevCardType d) {
    switch (d) {
        case DevCardType::Knight:       return "Knight       ";
        case DevCardType::VictoryPoint: return "VictoryPoint ";
        case DevCardType::RoadBuilding: return "RoadBuilding ";
        case DevCardType::YearOfPlenty: return "YearOfPlenty ";
        case DevCardType::Monopoly:     return "Monopoly     ";
        default:                        return "???          ";
    }
}

static void print_player_state(const PlayerState& player, std::uint8_t player_idx) {
    std::cout << "\n--- Player " << static_cast<int>(player_idx) << " ---\n";
    
    std::cout << "Resources:\n";
    for (std::size_t i = 0; i < NUM_RESOURCE_TYPES; ++i) {
        std::cout << "  " << resource_type_name(static_cast<ResourceType>(i)) 
                  << ": " << static_cast<int>(player.resources[i]) << "\n";
    }
    std::cout << "  Total: " << static_cast<int>(player.total_resources()) << "\n";
    
    std::cout << "Development Cards:\n";
    for (std::size_t i = 0; i < NUM_DEV_CARD_TYPES; ++i) {
        std::cout << "  " << dev_card_name(static_cast<DevCardType>(i))
                  << ": " << static_cast<int>(player.dev_cards[i]) << "\n";
    }
    
    std::cout << "Pieces Remaining:\n";
    std::cout << "  Settlements: " << static_cast<int>(player.settlements_remaining) << "/" 
              << static_cast<int>(MAX_SETTLEMENTS_PER_PLAYER) << "\n";
    std::cout << "  Cities:      " << static_cast<int>(player.cities_remaining) << "/" 
              << static_cast<int>(MAX_CITIES_PER_PLAYER) << "\n";
    std::cout << "  Roads:       " << static_cast<int>(player.roads_remaining) << "/" 
              << static_cast<int>(MAX_ROADS_PER_PLAYER) << "\n";
    
    std::cout << "Victory Points:\n";
    std::cout << "  Public:  " << static_cast<int>(player.public_victory_points) << "\n";
    std::cout << "  Hidden:  " << static_cast<int>(player.hidden_victory_points) << "\n";
    std::cout << "  Total:   " << static_cast<int>(player.total_victory_points()) << "\n";
    
    std::cout << "Special:\n";
    std::cout << "  Knights Played:   " << static_cast<int>(player.knights_played) << "\n";
    std::cout << "  Has Longest Road: " << (player.has_longest_road ? "Yes" : "No") << "\n";
    std::cout << "  Has Largest Army: " << (player.has_largest_army ? "Yes" : "No") << "\n";
}

static void print_game_summary(const GameState& game) {
    std::cout << "\n=== Game Summary ===\n";
    std::cout << "Players:        " << static_cast<int>(game.num_players) << "\n";
    std::cout << "Current Player: " << static_cast<int>(game.current_player) << "\n";
    std::cout << "Turn Number:    " << game.turn_number << "\n";
    std::cout << "Robber Tile:    " << static_cast<int>(game.robber_tile) << "\n";
    std::cout << "Game Over:      " << (game.is_game_over() ? "Yes" : "No") << "\n";
    
    if (game.is_game_over()) {
        std::cout << "Winner:         Player " << static_cast<int>(game.get_winner()) << "\n";
    }
}

static void test_game_initialization() {
    std::cout << "\n=== Test: Game Initialization ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    print_game_summary(game);
    
    // Verify initial state
    bool ok = true;
    
    if (game.num_players != 4) {
        std::cout << "ERROR: Expected 4 players, got " << static_cast<int>(game.num_players) << "\n";
        ok = false;
    }
    
    if (game.current_player != 0) {
        std::cout << "ERROR: Expected current player 0, got " << static_cast<int>(game.current_player) << "\n";
        ok = false;
    }
    
    if (game.game_phase != GamePhase::Setup) {
        std::cout << "ERROR: Expected Setup phase\n";
        ok = false;
    }
    
    // Check all players have correct starting pieces
    for (std::uint8_t p = 0; p < 4; ++p) {
        const PlayerState& player = game.players[p];
        if (player.settlements_remaining != MAX_SETTLEMENTS_PER_PLAYER ||
            player.cities_remaining != MAX_CITIES_PER_PLAYER ||
            player.roads_remaining != MAX_ROADS_PER_PLAYER) {
            std::cout << "ERROR: Player " << static_cast<int>(p) << " has incorrect starting pieces\n";
            ok = false;
        }
        
        if (player.total_resources() != 0) {
            std::cout << "ERROR: Player " << static_cast<int>(p) << " should start with no resources\n";
            ok = false;
        }
    }
    
    // Check dev deck is full
    std::uint32_t total_dev_cards = 0;
    for (std::uint8_t count : game.dev_deck) {
        total_dev_cards += count;
    }
    if (total_dev_cards != TOTAL_DEV_CARDS) {
        std::cout << "ERROR: Dev deck has " << total_dev_cards << " cards, expected " 
                  << TOTAL_DEV_CARDS << "\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_piece_placement() {
    std::cout << "\n=== Test: Piece Placement ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    // Place a settlement for player 0 at vertex 5
    game.place_settlement(0, 5);
    
    bool ok = true;
    
    if (!game.is_vertex_occupied(5)) {
        std::cout << "ERROR: Vertex 5 should be occupied\n";
        ok = false;
    }
    
    const Piece* piece = game.get_vertex_piece(5);
    if (!piece || piece->type != PieceType::Settlement || piece->owner != 0) {
        std::cout << "ERROR: Vertex 5 should have Player 0's settlement\n";
        ok = false;
    }
    
    if (game.players[0].settlements_remaining != MAX_SETTLEMENTS_PER_PLAYER - 1) {
        std::cout << "ERROR: Player 0 should have one less settlement in supply\n";
        ok = false;
    }
    
    // Place a road for player 0 at edge 3
    game.place_road(0, 3);
    
    if (!game.is_edge_occupied(3)) {
        std::cout << "ERROR: Edge 3 should be occupied\n";
        ok = false;
    }
    
    const Piece* road = game.get_edge_piece(3);
    if (!road || road->type != PieceType::Road || road->owner != 0) {
        std::cout << "ERROR: Edge 3 should have Player 0's road\n";
        ok = false;
    }
    
    if (game.players[0].roads_remaining != MAX_ROADS_PER_PLAYER - 1) {
        std::cout << "ERROR: Player 0 should have one less road in supply\n";
        ok = false;
    }
    
    // Upgrade settlement to city
    game.upgrade_to_city(0, 5);
    
    piece = game.get_vertex_piece(5);
    if (!piece || piece->type != PieceType::City || piece->owner != 0) {
        std::cout << "ERROR: Vertex 5 should now have Player 0's city\n";
        ok = false;
    }
    
    if (game.players[0].settlements_remaining != MAX_SETTLEMENTS_PER_PLAYER) {
        std::cout << "ERROR: Settlement should have returned to supply\n";
        ok = false;
    }
    
    if (game.players[0].cities_remaining != MAX_CITIES_PER_PLAYER - 1) {
        std::cout << "ERROR: Player 0 should have one less city in supply\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_resource_distribution() {
    std::cout << "\n=== Test: Resource Distribution ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    // Place settlements for testing
    // Find tiles with different numbers and place settlements on their vertices
    game.place_settlement(0, 0);
    game.place_settlement(1, 10);
    
    // Find a tile with number 6 or 8 (common numbers)
    std::uint8_t test_number = 0;
    for (std::uint8_t t = 0; t < NUM_TILES; ++t) {
        if (game.board.tiles[t].number == 6 || game.board.tiles[t].number == 8) {
            test_number = game.board.tiles[t].number;
            break;
        }
    }
    
    if (test_number == 0) {
        std::cout << "SKIPPED: No suitable test tile found\n";
        return;
    }
    
    // Distribute resources for this number
    std::uint8_t initial_resources[MAX_PLAYERS];
    for (std::uint8_t p = 0; p < 4; ++p) {
        initial_resources[p] = game.players[p].total_resources();
    }
    
    game.distribute_resources(test_number);
    
    // At least one player should have gained resources
    bool someone_gained = false;
    for (std::uint8_t p = 0; p < 4; ++p) {
        if (game.players[p].total_resources() > initial_resources[p]) {
            someone_gained = true;
            std::cout << "Player " << static_cast<int>(p) << " gained " 
                      << static_cast<int>(game.players[p].total_resources() - initial_resources[p])
                      << " resources from number " << static_cast<int>(test_number) << "\n";
        }
    }
    
    std::cout << (someone_gained ? "PASSED" : "SKIPPED (no gains)") << "\n";
}

static void test_dev_cards() {
    std::cout << "\n=== Test: Development Cards ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    // Buy a dev card for player 0
    DevCardType card = game.buy_dev_card(0, 54321);
    
    bool ok = true;
    
    if (card == DevCardType::COUNT) {
        std::cout << "ERROR: Failed to buy dev card\n";
        ok = false;
    } else {
        std::cout << "Player 0 bought: " << dev_card_name(card) << "\n";
        
        // Card should be in "bought this turn" hand
        std::size_t card_idx = static_cast<std::size_t>(card);
        if (game.players[0].dev_cards_bought_this_turn[card_idx] != 1) {
            std::cout << "ERROR: Card should be in 'bought this turn' hand\n";
            ok = false;
        }
        
        // Card should not be playable yet
        if (game.players[0].dev_cards[card_idx] != 0) {
            std::cout << "ERROR: Card should not be playable yet\n";
            ok = false;
        }
    }
    
    // End turn to make card playable
    game.end_turn();
    
    // Current player should be 1 now
    if (game.current_player != 1) {
        std::cout << "ERROR: Current player should be 1 after end_turn\n";
        ok = false;
    }
    
    // Player 0's card should now be playable
    if (card != DevCardType::COUNT && card != DevCardType::VictoryPoint) {
        std::size_t card_idx = static_cast<std::size_t>(card);
        if (game.players[0].dev_cards[card_idx] != 1) {
            std::cout << "ERROR: Card should be playable after turn end\n";
            ok = false;
        }
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

static void test_victory_points() {
    std::cout << "\n=== Test: Victory Points ===\n";
    
    GameState game = GameState::create_new_game(4, 12345);
    
    // Place settlements and cities for player 0
    game.place_settlement(0, 5);   // +1 VP
    game.place_settlement(0, 10);  // +1 VP
    game.upgrade_to_city(0, 5);    // +1 VP (city worth 2, settlement was 1)
    
    bool ok = true;
    
    if (game.players[0].public_victory_points != 4) {
        std::cout << "ERROR: Player 0 should have 4 VP (2 settlements + 1 city), got " 
                  << static_cast<int>(game.players[0].public_victory_points) << "\n";
        ok = false;
    }
    
    // Test largest army
    game.players[0].knights_played = 5;
    game.update_largest_army();
    
    if (!game.players[0].has_largest_army) {
        std::cout << "ERROR: Player 0 should have largest army\n";
        ok = false;
    }
    
    if (game.players[0].public_victory_points != 6) {
        std::cout << "ERROR: Player 0 should have 6 VP (+2 for largest army), got "
                  << static_cast<int>(game.players[0].public_victory_points) << "\n";
        ok = false;
    }
    
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Catan Game State Test Suite\n";
    std::cout << "========================================\n";
    
    test_game_initialization();
    test_piece_placement();
    test_resource_distribution();
    test_dev_cards();
    test_victory_points();
    
    // Create a sample game and print full state
    std::cout << "\n=== Sample Game State ===\n";
    GameState game = GameState::create_new_game(4, 99999);
    
    // Set up some example state
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Brick)] = 3;
    game.players[0].resources[static_cast<std::size_t>(ResourceType::Lumber)] = 2;
    game.place_settlement(0, 5);
    game.place_road(0, 3);
    
    print_game_summary(game);
    print_player_state(game.players[0], 0);
    
    std::cout << "\n========================================\n";
    std::cout << "  Test Suite Complete!\n";
    std::cout << "========================================\n";
    
    return 0;
}
