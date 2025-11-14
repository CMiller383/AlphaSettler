// move_gen.h
// Fast legal move generation for Catan RL engine.
// Critical for MCTS performance - called thousands of times per second.

#pragma once

#include <cstdint>
#include <vector>
#include "action.h"
#include "game_state.h"

namespace catan {

// Maximum possible actions in any state (for pre-allocation).
// Rough estimate: ~54 vertices + ~72 edges + ~25 trades + special actions
constexpr std::size_t MAX_ACTIONS_ESTIMATE = 200;

// Generate all legal actions for the current player in the given state.
// Actions are appended to the provided vector (does not clear it first).
// Returns number of actions generated.
std::size_t generate_legal_actions(const GameState& state, std::vector<Action>& out_actions);

// Specialized generators for different game phases/situations.
// These are called by generate_legal_actions but can also be used directly.

// Setup phase: generate legal initial settlement placements.
// Must satisfy distance rule (no adjacent settlements) and vertex availability.
std::size_t generate_initial_settlement_placements(const GameState& state, std::vector<Action>& out);

// Setup phase: generate legal initial road placements.
// Must be adjacent to the most recently placed settlement by current player.
std::size_t generate_initial_road_placements(const GameState& state, std::vector<Action>& out);

// Main game: generate legal settlement placements.
// Must satisfy: distance rule, vertex available, player has road connection, can afford.
std::size_t generate_settlement_placements(const GameState& state, std::uint8_t player_idx, 
                                           std::vector<Action>& out);

// Main game: generate legal road placements.
// Must satisfy: edge available, connected to player's road network, can afford.
std::size_t generate_road_placements(const GameState& state, std::uint8_t player_idx,
                                     std::vector<Action>& out);

// Main game: generate legal city upgrades.
// Must satisfy: vertex has player's settlement, player can afford.
std::size_t generate_city_upgrades(const GameState& state, std::uint8_t player_idx,
                                   std::vector<Action>& out);

// Generate all legal bank trades (4:1 trades).
// Player must have at least 4 of a resource to trade.
std::size_t generate_bank_trades(const GameState& state, std::uint8_t player_idx,
                                std::vector<Action>& out);

// Generate all legal port trades (3:1 or 2:1 trades based on harbor access).
// Player must have settlement/city adjacent to a harbor and enough resources.
std::size_t generate_port_trades(const GameState& state, std::uint8_t player_idx,
                                std::vector<Action>& out);

// Generate legal robber moves (after rolling 7 or playing knight).
// Can move to any tile except current robber position.
std::size_t generate_robber_moves(const GameState& state, std::vector<Action>& out);

// Generate legal steal targets (after moving robber).
// Can steal from any player with a settlement/city adjacent to robber tile.
std::size_t generate_steal_targets(const GameState& state, std::vector<Action>& out);

// Generate discard combinations (when player has >7 cards after rolling 7).
// Must discard exactly half (rounded down) of total cards.
std::size_t generate_discard_combinations(const GameState& state, std::uint8_t player_idx,
                                         std::vector<Action>& out);

// Development card play action generators.
// Generate legal actions for playing each type of development card.

// Generate knight play actions (move robber + optional steal).
std::size_t generate_knight_plays(const GameState& state, std::uint8_t player_idx,
                                  std::vector<Action>& out);

// Generate road building play actions (place 2 free roads).
std::size_t generate_road_building_plays(const GameState& state, std::uint8_t player_idx,
                                         std::vector<Action>& out);

// Generate year of plenty play actions (take 2 resources from bank).
std::size_t generate_year_of_plenty_plays(const GameState& state, std::uint8_t player_idx,
                                          std::vector<Action>& out);

// Generate monopoly play actions (take all of one resource type from opponents).
std::size_t generate_monopoly_plays(const GameState& state, std::uint8_t player_idx,
                                    std::vector<Action>& out);

// Helper functions for rule checking (not exported, but declared for testing).

// Check if a vertex satisfies the distance rule (no adjacent settlements).
bool check_distance_rule(const GameState& state, std::uint8_t vertex_idx);

// Check if a vertex is connected to player's road network.
// Used for settlement placement in main game.
bool is_connected_to_road_network(const GameState& state, std::uint8_t player_idx, 
                                  std::uint8_t vertex_idx);

// Check if an edge is connected to player's road network.
// An edge is connected if either endpoint has a player's road or settlement/city.
bool can_place_road(const GameState& state, std::uint8_t player_idx, std::uint8_t edge_idx);

// Find the most recently placed settlement by current player (for initial road placement).
std::uint8_t find_last_placed_settlement(const GameState& state, std::uint8_t player_idx);

} // namespace catan
