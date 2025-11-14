// state_transition.h
// Apply actions to game state - deterministic state transitions for MCTS.
// All functions mutate the provided GameState in-place for performance.

#pragma once

#include "game_state.h"
#include "action.h"
#include <cstdint>

namespace catan {

// Apply an action to the game state.
// Returns true if action was applied successfully, false if illegal.
// For MCTS: caller should ensure actions are legal (from generate_legal_actions).
// This function assumes legality and focuses on fast execution.
bool apply_action(GameState& state, const Action& action, std::uint32_t rng_seed = 0);

// Specialized transition functions (called by apply_action, can also be used directly).

// Setup phase: place initial settlement
void apply_initial_settlement(GameState& state, std::uint8_t vertex_idx);

// Setup phase: place initial road
void apply_initial_road(GameState& state, std::uint8_t edge_idx);

// Main game: place settlement (costs resources)
void apply_place_settlement(GameState& state, std::uint8_t player_idx, std::uint8_t vertex_idx);

// Main game: place road (costs resources)
void apply_place_road(GameState& state, std::uint8_t player_idx, std::uint8_t edge_idx);

// Main game: upgrade settlement to city (costs resources)
void apply_upgrade_city(GameState& state, std::uint8_t player_idx, std::uint8_t vertex_idx);

// Buy development card (costs resources, uses RNG to draw card)
void apply_buy_dev_card(GameState& state, std::uint8_t player_idx, std::uint32_t rng_seed);

// Play knight card (move robber, steal resource)
void apply_play_knight(GameState& state, std::uint8_t player_idx, 
                       std::uint8_t robber_tile, std::uint8_t steal_from);

// Play road building card (place 2 free roads)
void apply_play_road_building(GameState& state, std::uint8_t player_idx,
                               std::uint8_t edge1, std::uint8_t edge2);

// Play year of plenty card (take 2 resources from bank)
void apply_play_year_of_plenty(GameState& state, std::uint8_t player_idx,
                                ResourceType res1, ResourceType res2);

// Play monopoly card (take all of one resource from other players)
void apply_play_monopoly(GameState& state, std::uint8_t player_idx, ResourceType resource);

// Bank trade: give 4 of one resource, get 1 of another
void apply_bank_trade(GameState& state, std::uint8_t player_idx,
                      ResourceType give, ResourceType receive);

// Port trade: give N of one resource (2 or 3), get 1 of another
void apply_port_trade(GameState& state, std::uint8_t player_idx,
                      ResourceType give, ResourceType receive, std::uint8_t give_count);

// Move robber to a tile
void apply_move_robber(GameState& state, std::uint8_t tile_idx);

// Steal a random resource from a player
void apply_steal_from_player(GameState& state, std::uint8_t thief_idx, 
                              std::uint8_t victim_idx, std::uint32_t rng_seed);

// Discard resources (after rolling 7 with >7 cards)
void apply_discard(GameState& state, std::uint8_t player_idx,
                   const std::array<std::uint8_t, NUM_RESOURCE_TYPES>& discard_counts);

// Roll dice and handle consequences (resource distribution or robber)
void apply_roll_dice(GameState& state, std::uint32_t rng_seed);

// End current player's turn
void apply_end_turn(GameState& state);

// Helper: Check if setup phase is complete and transition to main game
void check_setup_complete(GameState& state);

// Helper: Advance to next player in setup phase (handles forward/reverse order)
void advance_setup_turn(GameState& state);

} // namespace catan
