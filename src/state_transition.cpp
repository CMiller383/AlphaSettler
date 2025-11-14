// state_transition.cpp
// Fast, deterministic state transitions for Catan RL engine.
// All functions mutate state in-place for MCTS performance.

#include "state_transition.h"
#include <random>
#include <algorithm>

namespace catan {

// ============================================================================
// Setup Phase Transitions
// ============================================================================

void apply_initial_settlement(GameState& state, std::uint8_t vertex_idx) {
    const std::uint8_t player_idx = state.current_player;
    
    // Place settlement (no cost in setup)
    state.place_settlement(player_idx, vertex_idx);
    
    // Track setup progress
    state.setup_settlements_placed[player_idx]++;
    
    // In second round of setup, player gets resources from adjacent tiles
    if (state.setup_settlements_placed[player_idx] == 2) {
        const Vertex& vertex = state.board.vertices[vertex_idx];
        for (std::uint8_t tile_idx : vertex.tiles) {
            if (tile_idx == INVALID_TILE) continue;
            
            const Tile& tile = state.board.tiles[tile_idx];
            if (tile.resource == Resource::Desert) continue;
            
            ResourceType res_type = tile_resource_to_player_resource(tile.resource);
            if (res_type != ResourceType::COUNT) {
                std::size_t res_idx = static_cast<std::size_t>(res_type);
                if (state.resource_bank[res_idx] > 0) {
                    state.players[player_idx].resources[res_idx]++;
                    state.resource_bank[res_idx]--;
                }
            }
        }
    }
}

void apply_initial_road(GameState& state, std::uint8_t edge_idx) {
    const std::uint8_t player_idx = state.current_player;
    
    // Place road (no cost in setup)
    state.place_road(player_idx, edge_idx);
    
    // Track setup progress
    state.setup_roads_placed[player_idx]++;
    
    // After placing road, check if player completed their pair
    advance_setup_turn(state);
}

void advance_setup_turn(GameState& state) {
    // Standard Catan setup:
    // Round 1: players 0,1,...,N-1 place settlement+road (forward order)
    // Round 2: players N-1,...,1,0 place settlement+road (reverse order)
    //
    // Key insight: Last player (N-1) places twice consecutively when transitioning from round 1 to round 2
    
    const std::uint8_t current = state.current_player;
    const std::uint8_t last_player = state.num_players - 1;
    
    const std::uint8_t s = state.setup_settlements_placed[current];
    const std::uint8_t r = state.setup_roads_placed[current];
    
    // Player must complete settlement+road pair before turn can advance
    bool completed_pair = (s == r && s > 0); // (1,1) or (2,2)
    
    if (!completed_pair) {
        // Player hasn't finished their current pair, stay on them
        return;
    }
    
    // Player just completed a pair. Now determine if/how to advance.
    
    // Determine which round we're in by checking if this was first or second pair
    if (s == 1) {
        // Just finished first pair (round 1)
        if (current < last_player) {
            // Not the last player, advance forward
            state.current_player++;
        }
        // else: last player just finished round 1, they stay to start round 2
    } else {
        // Just finished second pair (round 2)
        if (current > 0) {
            // Not player 0 yet, continue reverse order
            state.current_player--;
        } else {
            // Player 0 just finished - setup complete!
            state.game_phase = GamePhase::MainGame;
            state.turn_phase = TurnPhase::RollDice;
            state.current_player = 0;
        }
    }
}

void check_setup_complete(GameState& state) {
    // This function is now handled by advance_setup_turn
    // Kept for backward compatibility
    advance_setup_turn(state);
}

// ============================================================================
// Placement Transitions (Main Game)
// ============================================================================

void apply_place_settlement(GameState& state, std::uint8_t player_idx, std::uint8_t vertex_idx) {
    PlayerState& player = state.players[player_idx];
    
    // Pay cost and return resources to bank
    player.pay_resources(BuildCost::SETTLEMENT);
    state.return_resources_to_bank(BuildCost::SETTLEMENT);
    
    // Place settlement
    state.place_settlement(player_idx, vertex_idx);
}

void apply_place_road(GameState& state, std::uint8_t player_idx, std::uint8_t edge_idx) {
    PlayerState& player = state.players[player_idx];
    
    // Pay cost and return resources to bank
    player.pay_resources(BuildCost::ROAD);
    state.return_resources_to_bank(BuildCost::ROAD);
    
    // Place road
    state.place_road(player_idx, edge_idx);
    
    // Update longest road (expensive, but necessary)
    state.update_longest_road();
}

void apply_upgrade_city(GameState& state, std::uint8_t player_idx, std::uint8_t vertex_idx) {
    PlayerState& player = state.players[player_idx];
    
    // Pay cost and return resources to bank
    player.pay_resources(BuildCost::CITY);
    state.return_resources_to_bank(BuildCost::CITY);
    
    // Upgrade to city
    state.upgrade_to_city(player_idx, vertex_idx);
}

// ============================================================================
// Development Cards
// ============================================================================

void apply_buy_dev_card(GameState& state, std::uint8_t player_idx, std::uint32_t rng_seed) {
    PlayerState& player = state.players[player_idx];
    
    // Pay cost and return resources to bank
    player.pay_resources(BuildCost::DEV_CARD);
    state.return_resources_to_bank(BuildCost::DEV_CARD);
    
    // Buy card
    state.buy_dev_card(player_idx, rng_seed);
}

void apply_play_knight(GameState& state, std::uint8_t player_idx,
                       std::uint8_t robber_tile, std::uint8_t steal_from) {
    // Play the card
    state.play_knight(player_idx);
    
    // Move robber
    state.move_robber(robber_tile);
    
    // Steal resource if valid target
    if (steal_from < state.num_players && steal_from != player_idx) {
        PlayerState& victim = state.players[steal_from];
        if (victim.total_resources() > 0) {
            // Find a random resource to steal
            std::mt19937 rng(static_cast<std::uint32_t>(robber_tile) * 1000 + steal_from);
            std::uniform_int_distribution<std::size_t> dist(0, victim.total_resources() - 1);
            std::size_t steal_idx = dist(rng);
            
            // Find which resource type this corresponds to
            std::size_t cumulative = 0;
            for (std::size_t r = 0; r < NUM_RESOURCE_TYPES; ++r) {
                cumulative += victim.resources[r];
                if (steal_idx < cumulative && victim.resources[r] > 0) {
                    victim.resources[r]--;
                    state.players[player_idx].resources[r]++;
                    break;
                }
            }
        }
    }
}

void apply_play_road_building(GameState& state, std::uint8_t player_idx,
                               std::uint8_t edge1, std::uint8_t edge2) {
    PlayerState& player = state.players[player_idx];
    std::size_t rb_idx = static_cast<std::size_t>(DevCardType::RoadBuilding);
    
    if (player.dev_cards[rb_idx] == 0) return;
    
    // Remove card
    player.dev_cards[rb_idx]--;
    
    // Place roads (free)
    if (edge1 != INVALID_EDGE) {
        state.place_road(player_idx, edge1);
    }
    if (edge2 != INVALID_EDGE && edge2 != edge1) {
        state.place_road(player_idx, edge2);
    }
    
    state.update_longest_road();
}

void apply_play_year_of_plenty(GameState& state, std::uint8_t player_idx,
                                ResourceType res1, ResourceType res2) {
    PlayerState& player = state.players[player_idx];
    std::size_t yop_idx = static_cast<std::size_t>(DevCardType::YearOfPlenty);
    
    if (player.dev_cards[yop_idx] == 0) return;
    
    // Remove card
    player.dev_cards[yop_idx]--;
    
    // Gain resources from bank (check availability)
    if (res1 != ResourceType::COUNT) {
        std::size_t res1_idx = static_cast<std::size_t>(res1);
        if (state.resource_bank[res1_idx] > 0) {
            player.resources[res1_idx]++;
            state.resource_bank[res1_idx]--;
        }
    }
    if (res2 != ResourceType::COUNT) {
        std::size_t res2_idx = static_cast<std::size_t>(res2);
        if (state.resource_bank[res2_idx] > 0) {
            player.resources[res2_idx]++;
            state.resource_bank[res2_idx]--;
        }
    }
}

void apply_play_monopoly(GameState& state, std::uint8_t player_idx, ResourceType resource) {
    PlayerState& player = state.players[player_idx];
    std::size_t mono_idx = static_cast<std::size_t>(DevCardType::Monopoly);
    
    if (player.dev_cards[mono_idx] == 0) return;
    
    // Remove card
    player.dev_cards[mono_idx]--;
    
    // Take all of this resource from other players
    std::size_t res_idx = static_cast<std::size_t>(resource);
    for (std::uint8_t p = 0; p < state.num_players; ++p) {
        if (p == player_idx) continue;
        
        std::uint8_t stolen = state.players[p].resources[res_idx];
        state.players[p].resources[res_idx] = 0;
        player.resources[res_idx] += stolen;
    }
}

// ============================================================================
// Trading
// ============================================================================

void apply_bank_trade(GameState& state, std::uint8_t player_idx,
                      ResourceType give, ResourceType receive) {
    PlayerState& player = state.players[player_idx];
    
    std::size_t give_idx = static_cast<std::size_t>(give);
    std::size_t receive_idx = static_cast<std::size_t>(receive);
    
    // Trade 4:1 - check both player has resources and bank has resources to give
    if (player.resources[give_idx] >= 4 && state.resource_bank[receive_idx] > 0) {
        player.resources[give_idx] -= 4;
        state.resource_bank[give_idx] += 4;
        
        player.resources[receive_idx] += 1;
        state.resource_bank[receive_idx] -= 1;
    }
}

void apply_port_trade(GameState& state, std::uint8_t player_idx,
                      ResourceType give, ResourceType receive, std::uint8_t give_count) {
    PlayerState& player = state.players[player_idx];
    
    std::size_t give_idx = static_cast<std::size_t>(give);
    std::size_t receive_idx = static_cast<std::size_t>(receive);
    
    // Trade N:1 (typically 3:1 or 2:1) - check both player has resources and bank has resources to give
    if (player.resources[give_idx] >= give_count && state.resource_bank[receive_idx] > 0) {
        player.resources[give_idx] -= give_count;
        state.resource_bank[give_idx] += give_count;
        
        player.resources[receive_idx] += 1;
        state.resource_bank[receive_idx] -= 1;
    }
}

// ============================================================================
// Robber
// ============================================================================

void apply_move_robber(GameState& state, std::uint8_t tile_idx) {
    state.move_robber(tile_idx);
    
    // After moving robber, transition to steal phase or continue
    // (In full implementation, would check for steal targets)
    if (state.turn_phase == TurnPhase::Robber) {
        state.turn_phase = TurnPhase::Trading;
    }
}

void apply_steal_from_player(GameState& state, std::uint8_t thief_idx,
                              std::uint8_t victim_idx, std::uint32_t rng_seed) {
    if (victim_idx >= state.num_players || thief_idx >= state.num_players) return;
    if (victim_idx == thief_idx) return;
    
    PlayerState& victim = state.players[victim_idx];
    PlayerState& thief = state.players[thief_idx];
    
    if (victim.total_resources() == 0) return;
    
    // Pick a random resource to steal
    std::mt19937 rng(rng_seed);
    std::uniform_int_distribution<std::size_t> dist(0, victim.total_resources() - 1);
    std::size_t steal_idx = dist(rng);
    
    // Find which resource type this corresponds to
    std::size_t cumulative = 0;
    for (std::size_t r = 0; r < NUM_RESOURCE_TYPES; ++r) {
        cumulative += victim.resources[r];
        if (steal_idx < cumulative && victim.resources[r] > 0) {
            victim.resources[r]--;
            thief.resources[r]++;
            break;
        }
    }
}

void apply_discard(GameState& state, std::uint8_t player_idx,
                   const std::array<std::uint8_t, NUM_RESOURCE_TYPES>& discard_counts) {
    if (player_idx >= state.num_players) return;
    
    PlayerState& player = state.players[player_idx];
    
    // Remove discarded resources and return to bank
    for (std::size_t r = 0; r < NUM_RESOURCE_TYPES; ++r) {
        if (player.resources[r] >= discard_counts[r]) {
            player.resources[r] -= discard_counts[r];
            state.resource_bank[r] += discard_counts[r];
        }
    }
}

// ============================================================================
// Dice and Turn Management
// ============================================================================

void apply_roll_dice(GameState& state, std::uint32_t rng_seed) {
    // Roll dice
    std::uint8_t roll = state.roll_dice(rng_seed);
    
    // Handle based on roll
    if (roll == 7) {
        // Check if anyone needs to discard
        bool anyone_discarding = false;
        for (std::uint8_t p = 0; p < state.num_players; ++p) {
            if (state.players[p].total_resources() > 7) {
                anyone_discarding = true;
                break;
            }
        }
        
        if (anyone_discarding) {
            state.turn_phase = TurnPhase::Discard;
        } else {
            // Move to robber phase
            state.turn_phase = TurnPhase::Robber;
        }
    } else {
        // Resources already distributed by roll_dice()
        // Move to trading phase
        state.turn_phase = TurnPhase::Trading;
    }
}

void apply_end_turn(GameState& state) {
    state.end_turn();
    
    // Check for game over
    if (state.is_game_over()) {
        state.game_phase = GamePhase::Finished;
    }
}

// ============================================================================
// Main Action Dispatcher
// ============================================================================

bool apply_action(GameState& state, const Action& action, std::uint32_t rng_seed) {
    const std::uint8_t player_idx = state.current_player;
    
    switch (action.type) {
        case ActionType::PlaceInitialSettlement:
            apply_initial_settlement(state, action.location);
            break;
            
        case ActionType::PlaceInitialRoad:
            apply_initial_road(state, action.location);
            break;
            
        case ActionType::RollDice:
            apply_roll_dice(state, rng_seed);
            break;
            
        case ActionType::PlaceSettlement:
            apply_place_settlement(state, player_idx, action.location);
            break;
            
        case ActionType::PlaceRoad:
            apply_place_road(state, player_idx, action.location);
            break;
            
        case ActionType::UpgradeToCity:
            apply_upgrade_city(state, player_idx, action.location);
            break;
            
        case ActionType::BuyDevCard:
            apply_buy_dev_card(state, player_idx, rng_seed);
            break;
            
        case ActionType::PlayKnight:
            apply_play_knight(state, player_idx, action.location, action.param1);
            break;
            
        case ActionType::PlayRoadBuilding:
            apply_play_road_building(state, player_idx, action.param1, action.param2);
            break;
            
        case ActionType::PlayYearOfPlenty:
            apply_play_year_of_plenty(state, player_idx,
                static_cast<ResourceType>(action.param1),
                static_cast<ResourceType>(action.param2));
            break;
            
        case ActionType::PlayMonopoly:
            apply_play_monopoly(state, player_idx, static_cast<ResourceType>(action.param1));
            break;
            
        case ActionType::BankTrade:
            apply_bank_trade(state, player_idx,
                static_cast<ResourceType>(action.param1),
                static_cast<ResourceType>(action.param2));
            break;
            
        case ActionType::PortTrade:
            apply_port_trade(state, player_idx,
                static_cast<ResourceType>(action.param1),
                static_cast<ResourceType>(action.param2),
                action.param3);
            break;
            
        case ActionType::MoveRobber:
            apply_move_robber(state, action.location);
            break;
            
        case ActionType::StealFromPlayer:
            apply_steal_from_player(state, player_idx, action.param1, rng_seed);
            break;
            
        case ActionType::DiscardResources: {
            std::array<std::uint8_t, NUM_RESOURCE_TYPES> discards;
            discards[0] = action.param1;  // Brick
            discards[1] = action.param2;  // Lumber
            discards[2] = action.param3;  // Wool
            discards[3] = action.param4;  // Grain
            discards[4] = action.location; // Ore (reusing location field)
            apply_discard(state, player_idx, discards);
            break;
        }
            
        case ActionType::EndTurn:
            apply_end_turn(state);
            break;
            
        default:
            return false; // Unknown action type
    }
    
    return true;
}

} // namespace catan
