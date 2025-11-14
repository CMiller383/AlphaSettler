// game_state.h
// Complete Catan game state optimized for RL engine.
// Minimal allocation, plain structs, fast copying for MCTS simulations.

#pragma once

#include <cstdint>
#include <array>
#include "board_grid.h"
#include "player_state.h"
#include "dev_cards.h"

namespace catan {

// Game phases for state machine.
enum class GamePhase : std::uint8_t {
    Setup,              // Initial placement phase
    MainGame,           // Normal gameplay
    Finished            // Game has ended
};

// Sub-phases within a turn.
enum class TurnPhase : std::uint8_t {
    RollDice,           // Player must roll dice
    Robber,             // Player must move robber (after rolling 7)
    Discard,            // Players with >7 cards must discard (after rolling 7)
    Trading,            // Player can trade and build
    EndTurn             // Turn is complete, advance to next player
};

// Piece types that can be placed on the board.
enum class PieceType : std::uint8_t {
    None = 0,
    Road,
    Settlement,
    City
};

// Represents a piece placed on the board.
// For roads: placed on edges
// For settlements/cities: placed on vertices
struct Piece {
    PieceType type{PieceType::None};
    std::uint8_t owner{0xFF};  // Player index (0xFF = no owner)
};

// Complete game state.
// Designed for fast copying (needed for MCTS tree search).
struct GameState {
    // Board layout (tiles, vertices, edges).
    BoardGrid board;

    // Per-player state (resources, cards, pieces).
    std::array<PlayerState, MAX_PLAYERS> players{};

    // Number of active players (2-4 in standard Catan).
    std::uint8_t num_players{4};

    // Current player index (0 to num_players-1).
    std::uint8_t current_player{0};

    // Current game phase and turn phase.
    GamePhase game_phase{GamePhase::Setup};
    TurnPhase turn_phase{TurnPhase::RollDice};

    // Setup phase tracking: how many settlements/roads each player has placed.
    std::array<std::uint8_t, MAX_PLAYERS> setup_settlements_placed{};
    std::array<std::uint8_t, MAX_PLAYERS> setup_roads_placed{};

    // Pieces placed on the board.
    // Vertices: settlements or cities
    // Edges: roads
    std::array<Piece, NUM_VERTICES> vertex_pieces{};
    std::array<Piece, NUM_EDGES> edge_pieces{};

    // Robber position (tile index).
    std::uint8_t robber_tile{0xFF};

    // Development card deck (cards remaining to buy).
    std::array<std::uint8_t, NUM_DEV_CARD_TYPES> dev_deck{};

    // Resource bank (standard Catan has 19 of each resource type)
    std::array<std::uint8_t, NUM_RESOURCE_TYPES> resource_bank{};

    // Longest road and largest army tracking.
    std::uint8_t longest_road_owner{0xFF};   // 0xFF = no owner yet
    std::uint8_t longest_road_length{4};      // Minimum length to claim is 5
    std::uint8_t largest_army_owner{0xFF};    // 0xFF = no owner yet
    std::uint8_t largest_army_count{2};       // Minimum knights to claim is 3

    // Turn counter (for debugging and analytics).
    std::uint32_t turn_number{0};

    // Last dice roll (for tracking/debugging).
    std::uint8_t last_dice_roll{0};

    // Initialize a new game with given number of players and board seed.
    static GameState create_new_game(std::uint8_t num_players, std::uint32_t board_seed);

    // Check if game is over (someone reached 10 VP).
    bool is_game_over() const;

    // Get winner (only valid if is_game_over() returns true).
    std::uint8_t get_winner() const;

    // Check if a vertex is occupied.
    bool is_vertex_occupied(std::uint8_t vertex_idx) const {
        return vertex_pieces[vertex_idx].type != PieceType::None;
    }

    // Check if an edge is occupied.
    bool is_edge_occupied(std::uint8_t edge_idx) const {
        return edge_pieces[edge_idx].type != PieceType::None;
    }

    // Get piece at vertex (nullptr if empty).
    const Piece* get_vertex_piece(std::uint8_t vertex_idx) const {
        if (vertex_idx >= NUM_VERTICES) return nullptr;
        const Piece& p = vertex_pieces[vertex_idx];
        return (p.type == PieceType::None) ? nullptr : &p;
    }

    // Get piece at edge (nullptr if empty).
    const Piece* get_edge_piece(std::uint8_t edge_idx) const {
        if (edge_idx >= NUM_EDGES) return nullptr;
        const Piece& p = edge_pieces[edge_idx];
        return (p.type == PieceType::None) ? nullptr : &p;
    }

    // Place a settlement at a vertex (caller must validate legality).
    void place_settlement(std::uint8_t player_idx, std::uint8_t vertex_idx);

    // Upgrade a settlement to a city (caller must validate legality).
    void upgrade_to_city(std::uint8_t player_idx, std::uint8_t vertex_idx);

    // Place a road at an edge (caller must validate legality).
    void place_road(std::uint8_t player_idx, std::uint8_t edge_idx);

    // Roll dice and distribute resources.
    // Returns the dice value rolled.
    std::uint8_t roll_dice(std::uint32_t seed);

    // Distribute resources for a given dice roll (helper for deterministic testing).
    void distribute_resources(std::uint8_t dice_value);

    // Return resources to the bank (used when spending resources).
    void return_resources_to_bank(const std::array<std::uint8_t, NUM_RESOURCE_TYPES>& cost) {
        for (std::size_t i = 0; i < NUM_RESOURCE_TYPES; ++i) {
            resource_bank[i] += cost[i];
        }
    }

    // Move robber to a tile.
    void move_robber(std::uint8_t tile_idx);

    // Buy a development card (caller must validate affordability and deck availability).
    // Returns the card type drawn.
    DevCardType buy_dev_card(std::uint8_t player_idx, std::uint32_t seed);

    // Play a knight card.
    void play_knight(std::uint8_t player_idx);

    // End current player's turn and advance to next player.
    void end_turn();

    // Recalculate longest road for all players (expensive, call only when roads change).
    void update_longest_road();

    // Recalculate largest army (call when knights are played).
    void update_largest_army();

    // Recalculate victory points for a player.
    void update_victory_points(std::uint8_t player_idx);
    
    // Check if player has access to a specific harbor type.
    // Returns true if player has settlement/city adjacent to a harbor of given type.
    bool has_harbor_access(std::uint8_t player_idx, HarborType harbor_type) const;
    
    // Get best trade ratio for a resource (considers harbors).
    // Returns 2, 3, or 4 (representing 2:1, 3:1, or 4:1 trade).
    std::uint8_t get_trade_ratio(std::uint8_t player_idx, ResourceType resource) const;
};

} // namespace catan
