// move_gen.cpp
// Fast legal move generation for Catan RL self-play.
// Optimized for MCTS: minimal allocations, fast rule checks.

#include "move_gen.h"
#include <algorithm>
#include <queue>
#include <array>
#include <random>

namespace catan {

// ============================================================================
// Helper: Distance Rule
// ============================================================================

bool check_distance_rule(const GameState& state, std::uint8_t vertex_idx) {
    if (vertex_idx >= NUM_VERTICES) return false;

    // Check if vertex itself is occupied
    if (state.is_vertex_occupied(vertex_idx)) return false;

    // Check all adjacent vertices (via edges)
    const Vertex& vertex = state.board.vertices[vertex_idx];
    for (std::uint8_t edge_idx : vertex.edges) {
        if (edge_idx == INVALID_EDGE) continue;

        const Edge& edge = state.board.edges[edge_idx];
        // Check the other endpoint of this edge
        for (std::uint8_t other_vertex : edge.vertices) {
            if (other_vertex == vertex_idx) continue;
            if (other_vertex == INVALID_VERTEX) continue;
            if (state.is_vertex_occupied(other_vertex)) {
                return false; // Adjacent vertex is occupied
            }
        }
    }

    return true;
}

// ============================================================================
// Helper: Road Network Connectivity
// ============================================================================

bool is_connected_to_road_network(const GameState& state, std::uint8_t player_idx, 
                                  std::uint8_t vertex_idx) {
    if (vertex_idx >= NUM_VERTICES) return false;

    // A vertex is connected to the player's road network if the player has a road on any adjacent edge
    const Vertex& vertex = state.board.vertices[vertex_idx];
    for (std::uint8_t edge_idx : vertex.edges) {
        if (edge_idx == INVALID_EDGE) continue;
        
        const Piece& edge_piece = state.edge_pieces[edge_idx];
        if (edge_piece.type == PieceType::Road && edge_piece.owner == player_idx) {
            return true;
        }
    }

    return false;
}

bool can_place_road(const GameState& state, std::uint8_t player_idx, std::uint8_t edge_idx) {
    if (edge_idx >= NUM_EDGES) return false;

    // Edge must be unoccupied
    if (state.is_edge_occupied(edge_idx)) return false;

    // Edge must connect to player's road network at one of its endpoints
    const Edge& edge = state.board.edges[edge_idx];
    
    for (std::uint8_t endpoint_idx : edge.vertices) {
        if (endpoint_idx == INVALID_VERTEX) continue;
        
        // Check if this endpoint is connected to player's network
        if (is_connected_to_road_network(state, player_idx, endpoint_idx)) {
            return true;
        }
    }

    return false;
}

// ============================================================================
// Helper: Find Last Placed Settlement (for initial road placement)
// ============================================================================

std::uint8_t find_last_placed_settlement(const GameState& state, std::uint8_t player_idx) {
    // In setup phase, find the settlement just placed by this player
    // We look for the most recent settlement owned by player_idx
    
    // Simple approach: count how many settlements this player has placed
    // and find the expected vertex based on setup phase tracking
    std::uint8_t count = state.setup_settlements_placed[player_idx];
    
    if (count == 0) return INVALID_VERTEX;

    // Scan vertices to find player's settlements
    // Return the last one found (this is not perfect, but works for setup)
    std::uint8_t last_vertex = INVALID_VERTEX;
    for (std::uint8_t v = 0; v < NUM_VERTICES; ++v) {
        const Piece& piece = state.vertex_pieces[v];
        if (piece.type == PieceType::Settlement && piece.owner == player_idx) {
            last_vertex = v;
        }
    }
    
    return last_vertex;
}

// ============================================================================
// Setup Phase Generators
// ============================================================================

std::size_t generate_initial_settlement_placements(const GameState& state, 
                                                   std::vector<Action>& out) {
    std::size_t initial_size = out.size();

    // All vertices that satisfy distance rule
    for (std::uint8_t v = 0; v < NUM_VERTICES; ++v) {
        if (check_distance_rule(state, v)) {
            out.push_back(Action::place_initial_settlement(v));
        }
    }
    
    // CRITICAL FIX: Shuffle initial placements to prevent sequential bias!
    // Without shuffling, MCTS with few simulations tends to pick low-index vertices
    // (vertex 0, 1, 2, ...) leading to poor placements
    std::size_t added = out.size() - initial_size;
    if (added > 1) {
        // Shuffle only the newly added actions using thread-local RNG
        static thread_local std::mt19937 rng{std::random_device{}()};
        std::shuffle(out.begin() + initial_size, out.end(), rng);
    }

    return added;
}

std::size_t generate_initial_road_placements(const GameState& state, 
                                             std::vector<Action>& out) {
    std::size_t initial_size = out.size();

    // Find last placed settlement by current player
    std::uint8_t settlement_vertex = find_last_placed_settlement(state, state.current_player);
    if (settlement_vertex == INVALID_VERTEX) return 0;

    // Find all edges adjacent to this vertex
    const Vertex& vertex = state.board.vertices[settlement_vertex];
    for (std::uint8_t edge_idx : vertex.edges) {
        if (edge_idx == INVALID_EDGE) continue;
        
        // Edge must be unoccupied
        if (!state.is_edge_occupied(edge_idx)) {
            out.push_back(Action::place_initial_road(edge_idx));
        }
    }

    return out.size() - initial_size;
}

// ============================================================================
// Main Game Generators
// ============================================================================

std::size_t generate_settlement_placements(const GameState& state, std::uint8_t player_idx,
                                          std::vector<Action>& out) {
    if (player_idx >= state.num_players) return 0;
    
    std::size_t initial_size = out.size();
    const PlayerState& player = state.players[player_idx];

    // Check affordability
    if (!player.can_afford(BuildCost::SETTLEMENT)) return 0;
    
    // Check if player has settlements remaining
    if (player.settlements_remaining == 0) return 0;

    // Find all vertices that satisfy:
    // 1. Distance rule
    // 2. Connected to player's road network
    for (std::uint8_t v = 0; v < NUM_VERTICES; ++v) {
        if (check_distance_rule(state, v) && 
            is_connected_to_road_network(state, player_idx, v)) {
            out.push_back(Action::place_settlement(v));
        }
    }

    return out.size() - initial_size;
}

std::size_t generate_road_placements(const GameState& state, std::uint8_t player_idx,
                                     std::vector<Action>& out) {
    if (player_idx >= state.num_players) return 0;
    
    std::size_t initial_size = out.size();
    const PlayerState& player = state.players[player_idx];

    // Check affordability
    if (!player.can_afford(BuildCost::ROAD)) return 0;
    
    // Check if player has roads remaining
    if (player.roads_remaining == 0) return 0;

    // Find all edges that can be built on
    for (std::uint8_t e = 0; e < NUM_EDGES; ++e) {
        if (can_place_road(state, player_idx, e)) {
            out.push_back(Action::place_road(e));
        }
    }

    return out.size() - initial_size;
}

std::size_t generate_city_upgrades(const GameState& state, std::uint8_t player_idx,
                                   std::vector<Action>& out) {
    if (player_idx >= state.num_players) return 0;
    
    std::size_t initial_size = out.size();
    const PlayerState& player = state.players[player_idx];

    // Check affordability
    if (!player.can_afford(BuildCost::CITY)) return 0;
    
    // Check if player has cities remaining
    if (player.cities_remaining == 0) return 0;

    // Find all settlements owned by player
    for (std::uint8_t v = 0; v < NUM_VERTICES; ++v) {
        const Piece& piece = state.vertex_pieces[v];
        if (piece.type == PieceType::Settlement && piece.owner == player_idx) {
            out.push_back(Action::upgrade_to_city(v));
        }
    }

    return out.size() - initial_size;
}

// ============================================================================
// Trading and Purchasing
// ============================================================================

std::size_t generate_bank_trades(const GameState& state, std::uint8_t player_idx,
                                std::vector<Action>& out) {
    if (player_idx >= state.num_players) return 0;
    
    std::size_t initial_size = out.size();
    const PlayerState& player = state.players[player_idx];

    // For each resource type player has at least 4 of
    for (std::size_t give_idx = 0; give_idx < NUM_RESOURCE_TYPES; ++give_idx) {
        if (player.resources[give_idx] < 4) continue;

        ResourceType give = static_cast<ResourceType>(give_idx);

        // Can trade for any other resource type
        for (std::size_t receive_idx = 0; receive_idx < NUM_RESOURCE_TYPES; ++receive_idx) {
            if (receive_idx == give_idx) continue; // Can't trade for same resource

            ResourceType receive = static_cast<ResourceType>(receive_idx);
            out.push_back(Action::bank_trade(give, receive));
        }
    }

    return out.size() - initial_size;
}

std::size_t generate_port_trades(const GameState& state, std::uint8_t player_idx,
                                 std::vector<Action>& out) {
    if (player_idx >= state.num_players) return 0;
    
    std::size_t initial_size = out.size();
    const PlayerState& player = state.players[player_idx];
    
    // For each resource type, check if player has access to a better trade ratio
    for (std::size_t give_idx = 0; give_idx < NUM_RESOURCE_TYPES; ++give_idx) {
        ResourceType give = static_cast<ResourceType>(give_idx);
        std::uint8_t trade_ratio = state.get_trade_ratio(player_idx, give);
        
        // Only generate port trades if better than 4:1 (handled by bank trades)
        if (trade_ratio >= 4 || player.resources[give_idx] < trade_ratio) {
            continue;
        }
        
        // Can trade for any other resource type
        for (std::size_t receive_idx = 0; receive_idx < NUM_RESOURCE_TYPES; ++receive_idx) {
            if (receive_idx == give_idx) continue; // Can't trade for same resource
            
            ResourceType receive = static_cast<ResourceType>(receive_idx);
            out.push_back(Action::port_trade(give, receive, trade_ratio));
        }
    }
    
    return out.size() - initial_size;
}

// ============================================================================
// Robber Actions
// ============================================================================

std::size_t generate_robber_moves(const GameState& state, std::vector<Action>& out) {
    std::size_t initial_size = out.size();

    // Can move robber to any tile except current position
    for (std::uint8_t t = 0; t < NUM_TILES; ++t) {
        if (t != state.robber_tile) {
            out.push_back(Action::move_robber(t));
        }
    }

    return out.size() - initial_size;
}

std::size_t generate_steal_targets(const GameState& state, std::vector<Action>& out) {
    std::size_t initial_size = out.size();

    if (state.robber_tile >= NUM_TILES) return 0;

    // Find all players with settlements/cities adjacent to robber tile
    std::array<bool, MAX_PLAYERS> can_steal_from{false, false, false, false};

    // Get vertices of robber tile
    std::uint8_t vertices[6];
    state.board.get_tile_vertices(state.robber_tile, vertices);

    for (std::uint8_t v_idx : vertices) {
        const Piece& piece = state.vertex_pieces[v_idx];
        if (piece.type != PieceType::None && 
            piece.owner != state.current_player &&
            piece.owner < state.num_players) {
            
            // Check if this player has any resources
            if (state.players[piece.owner].total_resources() > 0) {
                can_steal_from[piece.owner] = true;
            }
        }
    }

    // Generate steal actions
    for (std::uint8_t p = 0; p < state.num_players; ++p) {
        if (can_steal_from[p]) {
            out.push_back(Action::steal_from_player(p));
        }
    }

    return out.size() - initial_size;
}

// ============================================================================
// Discard Actions
// ============================================================================

// Recursive helper to generate all combinations that sum to target
static void generate_discard_recursive(
    const std::array<std::uint8_t, NUM_RESOURCE_TYPES>& available,
    std::size_t resource_idx,
    std::uint8_t remaining_to_discard,
    std::array<std::uint8_t, NUM_RESOURCE_TYPES>& current,
    std::vector<Action>& out)
{
    // Base case: all resources chosen
    if (resource_idx >= NUM_RESOURCE_TYPES) {
        if (remaining_to_discard == 0) {
            out.push_back(Action::discard_resources(
                current[0], current[1], current[2], current[3], current[4]
            ));
        }
        return;
    }

    // Try all possible counts for this resource
    std::uint8_t max_discard = std::min(available[resource_idx], remaining_to_discard);
    for (std::uint8_t count = 0; count <= max_discard; ++count) {
        current[resource_idx] = count;
        generate_discard_recursive(available, resource_idx + 1, 
                                   remaining_to_discard - count, current, out);
    }
}

std::size_t generate_discard_combinations(const GameState& state, std::uint8_t player_idx,
                                         std::vector<Action>& out) {
    if (player_idx >= state.num_players) return 0;
    
    std::size_t initial_size = out.size();
    const PlayerState& player = state.players[player_idx];

    std::uint8_t total = player.total_resources();
    if (total <= 7) return 0; // No need to discard

    std::uint8_t to_discard = total / 2;

    // Generate all combinations recursively
    std::array<std::uint8_t, NUM_RESOURCE_TYPES> current{};
    generate_discard_recursive(player.resources, 0, to_discard, current, out);

    return out.size() - initial_size;
}

// ============================================================================
// Development Card Play Actions
// ============================================================================

std::size_t generate_knight_plays(const GameState& state, std::uint8_t player_idx,
                                   std::vector<Action>& out) {
    std::size_t initial_size = out.size();

    const PlayerState& player = state.players[player_idx];
    std::size_t knight_idx = static_cast<std::size_t>(DevCardType::Knight);
    
    if (player.dev_cards[knight_idx] == 0) return 0;

    // Generate all possible robber moves (excluding current robber tile)
    for (std::uint8_t tile_idx = 0; tile_idx < NUM_TILES; ++tile_idx) {
        if (tile_idx == state.robber_tile) continue;

        // For each robber placement, generate steal options
        // Get vertices of this tile to find potential steal targets
        std::uint8_t vertices[6];
        state.board.get_tile_vertices(tile_idx, vertices);
        
        std::array<bool, MAX_PLAYERS> can_steal{};
        for (std::uint8_t v_idx : vertices) {
            const Piece& piece = state.vertex_pieces[v_idx];
            if (piece.type == PieceType::Settlement || piece.type == PieceType::City) {
                if (piece.owner != player_idx && piece.owner < state.num_players) {
                    can_steal[piece.owner] = true;
                }
            }
        }

        // Generate action for each stealable player (or 0xFF for no steal)
        bool has_steal_target = false;
        for (std::uint8_t victim = 0; victim < state.num_players; ++victim) {
            if (can_steal[victim] && state.players[victim].total_resources() > 0) {
                out.push_back(Action::play_knight(tile_idx, victim));
                has_steal_target = true;
            }
        }

        // If no valid steal targets, still allow knight play (robber move only)
        if (!has_steal_target) {
            out.push_back(Action::play_knight(tile_idx, 0xFF));
        }
    }

    return out.size() - initial_size;
}

std::size_t generate_road_building_plays(const GameState& state, std::uint8_t player_idx,
                                          std::vector<Action>& out) {
    std::size_t initial_size = out.size();

    const PlayerState& player = state.players[player_idx];
    std::size_t rb_idx = static_cast<std::size_t>(DevCardType::RoadBuilding);
    
    if (player.dev_cards[rb_idx] == 0) return 0;
    if (player.roads_remaining < 1) return 0;

    // Generate all valid first road placements
    std::vector<std::uint8_t> first_roads;
    for (std::uint8_t e = 0; e < NUM_EDGES; ++e) {
        if (can_place_road(state, player_idx, e)) {
            first_roads.push_back(e);
        }
    }

    // For each first road, generate second road options
    for (std::uint8_t edge1 : first_roads) {
        // Simulate placing first road to check connectivity for second
        // (This is approximation - full simulation would require state copy)
        // For simplicity, check second roads that are valid NOW
        for (std::uint8_t edge2 = 0; edge2 < NUM_EDGES; ++edge2) {
            if (edge2 == edge1) continue;
            if (can_place_road(state, player_idx, edge2)) {
                out.push_back(Action::play_road_building(edge1, edge2));
            }
        }

        // Also allow playing just one road (if player only has 1 road left)
        if (player.roads_remaining == 1) {
            out.push_back(Action::play_road_building(edge1, INVALID_EDGE));
        }
    }

    return out.size() - initial_size;
}

std::size_t generate_year_of_plenty_plays(const GameState& state, std::uint8_t player_idx,
                                           std::vector<Action>& out) {
    std::size_t initial_size = out.size();

    const PlayerState& player = state.players[player_idx];
    std::size_t yop_idx = static_cast<std::size_t>(DevCardType::YearOfPlenty);
    
    if (player.dev_cards[yop_idx] == 0) return 0;

    // Generate all pairs of resources (with replacement)
    for (std::uint8_t r1 = 0; r1 < NUM_RESOURCE_TYPES; ++r1) {
        // Check bank has this resource
        if (state.resource_bank[r1] == 0) continue;
        
        for (std::uint8_t r2 = 0; r2 < NUM_RESOURCE_TYPES; ++r2) {
            // Check bank has second resource (or same resource has >= 2)
            if (r1 == r2 && state.resource_bank[r2] < 2) continue;
            if (r1 != r2 && state.resource_bank[r2] == 0) continue;
            
            out.push_back(Action::play_year_of_plenty(
                static_cast<ResourceType>(r1),
                static_cast<ResourceType>(r2)
            ));
        }
    }

    return out.size() - initial_size;
}

std::size_t generate_monopoly_plays(const GameState& state, std::uint8_t player_idx,
                                     std::vector<Action>& out) {
    std::size_t initial_size = out.size();

    const PlayerState& player = state.players[player_idx];
    std::size_t mono_idx = static_cast<std::size_t>(DevCardType::Monopoly);
    
    if (player.dev_cards[mono_idx] == 0) return 0;

    // Generate action for each resource type
    for (std::uint8_t r = 0; r < NUM_RESOURCE_TYPES; ++r) {
        out.push_back(Action::play_monopoly(static_cast<ResourceType>(r)));
    }

    return out.size() - initial_size;
}

// ============================================================================
// Main Legal Action Generator
// ============================================================================

std::size_t generate_legal_actions(const GameState& state, std::vector<Action>& out_actions) {
    std::size_t initial_size = out_actions.size();

    const std::uint8_t player_idx = state.current_player;

    // Setup phase
    if (state.game_phase == GamePhase::Setup) {
        std::uint8_t settlements_placed = state.setup_settlements_placed[player_idx];
        std::uint8_t roads_placed = state.setup_roads_placed[player_idx];

        // Each player places 2 settlements + 2 roads total (one pair per round)
        // Valid states during a turn: (0,0), (1,0), (1,1), (2,1), (2,2)
        // After (1,1) or (2,2), turn advances to next player
        
        if (settlements_placed > roads_placed) {
            // Need to place road for last settlement
            generate_initial_road_placements(state, out_actions);
        } else if (settlements_placed == roads_placed && settlements_placed < 2) {
            // Pair is complete but need another pair - place next settlement
            generate_initial_settlement_placements(state, out_actions);
        }
        // else: both conditions false means (2,2) complete - no legal moves
        
        return out_actions.size() - initial_size;
    }

    // Main game phase
    if (state.game_phase == GamePhase::MainGame) {
        switch (state.turn_phase) {
            case TurnPhase::RollDice:
                // Only action is to roll dice
                out_actions.push_back(Action::roll_dice());
                break;

            case TurnPhase::Discard:
                // Players with >7 cards must discard
                if (state.players[player_idx].total_resources() > 7) {
                    generate_discard_combinations(state, player_idx, out_actions);
                } else {
                    // This player doesn't need to discard, move to next player
                    // (In full implementation, would check all players)
                    out_actions.push_back(Action::end_turn()); // Placeholder
                }
                break;

            case TurnPhase::Robber:
                // Must move robber after rolling 7
                generate_robber_moves(state, out_actions);
                break;

            case TurnPhase::Trading: {
                // Player can build, trade, play dev cards, or end turn
                generate_settlement_placements(state, player_idx, out_actions);
                generate_road_placements(state, player_idx, out_actions);
                generate_city_upgrades(state, player_idx, out_actions);
                generate_bank_trades(state, player_idx, out_actions);
                generate_port_trades(state, player_idx, out_actions);

                // Can buy dev card if affordable and deck not empty
                const PlayerState& player = state.players[player_idx];
                if (player.can_afford(BuildCost::DEV_CARD)) {
                    std::uint32_t total_dev_cards = 0;
                    for (std::uint8_t count : state.dev_deck) {
                        total_dev_cards += count;
                    }
                    if (total_dev_cards > 0) {
                        out_actions.push_back(Action::buy_dev_card());
                    }
                }

                // Generate dev card play actions
                generate_knight_plays(state, player_idx, out_actions);
                generate_road_building_plays(state, player_idx, out_actions);
                generate_year_of_plenty_plays(state, player_idx, out_actions);
                generate_monopoly_plays(state, player_idx, out_actions);
                
                // Can always end turn
                out_actions.push_back(Action::end_turn());
                break;
            }

            default:
                break;
        }
    }

    return out_actions.size() - initial_size;
}

} // namespace catan
