// game_state.cpp
// Implementation of Catan game state optimized for RL self-play.

#include "game_state.h"
#include <algorithm>
#include <random>
#include <queue>
#include <vector>

namespace catan {

GameState GameState::create_new_game(std::uint8_t num_players, std::uint32_t board_seed) {
    GameState state;

    // Validate player count
    state.num_players = std::min(std::max(num_players, static_cast<std::uint8_t>(2)), 
                                  static_cast<std::uint8_t>(MAX_PLAYERS));

    // Initialize board
    state.board = BoardGrid::create_default();
    state.board.randomize(board_seed);

    // Initialize development card deck
    for (std::size_t i = 0; i < NUM_DEV_CARD_TYPES; ++i) {
        state.dev_deck[i] = STANDARD_DEV_CARD_COUNTS[i];
    }

    // Initialize resource bank (19 of each resource type in standard Catan)
    for (std::size_t i = 0; i < NUM_RESOURCE_TYPES; ++i) {
        state.resource_bank[i] = 19;
    }

    // Place robber on desert tile initially
    for (std::uint8_t t = 0; t < NUM_TILES; ++t) {
        if (state.board.tiles[t].resource == Resource::Desert) {
            state.robber_tile = t;
            break;
        }
    }

    // Initialize player states (defaults from PlayerState are sufficient)
    // All players start with zero resources and pieces in supply

    // Initialize pieces arrays to empty
    for (auto& p : state.vertex_pieces) {
        p.type = PieceType::None;
        p.owner = 0xFF;
    }
    for (auto& p : state.edge_pieces) {
        p.type = PieceType::None;
        p.owner = 0xFF;
    }

    // Start in setup phase
    state.game_phase = GamePhase::Setup;
    state.turn_phase = TurnPhase::RollDice;  // Not used in setup, but set for consistency
    state.current_player = 0;
    state.turn_number = 0;

    return state;
}

bool GameState::is_game_over() const {
    if (game_phase != GamePhase::MainGame) return false;
    
    for (std::uint8_t p = 0; p < num_players; ++p) {
        if (players[p].total_victory_points() >= 10) {
            return true;
        }
    }
    return false;
}

std::uint8_t GameState::get_winner() const {
    std::uint8_t winner = 0xFF;
    std::uint8_t max_vp = 0;
    
    for (std::uint8_t p = 0; p < num_players; ++p) {
        std::uint8_t vp = players[p].total_victory_points();
        if (vp > max_vp) {
            max_vp = vp;
            winner = p;
        }
    }
    
    return winner;
}

void GameState::place_settlement(std::uint8_t player_idx, std::uint8_t vertex_idx) {
    if (player_idx >= num_players || vertex_idx >= NUM_VERTICES) return;
    if (is_vertex_occupied(vertex_idx)) return;

    // Place the piece
    vertex_pieces[vertex_idx].type = PieceType::Settlement;
    vertex_pieces[vertex_idx].owner = player_idx;

    // Update player state
    PlayerState& player = players[player_idx];
    if (player.settlements_remaining > 0) {
        player.settlements_remaining--;
    }

    // Update victory points (settlements are worth 1 VP)
    update_victory_points(player_idx);
}

void GameState::upgrade_to_city(std::uint8_t player_idx, std::uint8_t vertex_idx) {
    if (player_idx >= num_players || vertex_idx >= NUM_VERTICES) return;

    Piece& piece = vertex_pieces[vertex_idx];
    if (piece.type != PieceType::Settlement || piece.owner != player_idx) return;

    // Upgrade the piece
    piece.type = PieceType::City;

    // Update player state (settlement returns to supply, city is used)
    PlayerState& player = players[player_idx];
    player.settlements_remaining++;
    if (player.cities_remaining > 0) {
        player.cities_remaining--;
    }

    // Update victory points (cities are worth 2 VP, settlements 1 VP, so net +1)
    update_victory_points(player_idx);
}

void GameState::place_road(std::uint8_t player_idx, std::uint8_t edge_idx) {
    if (player_idx >= num_players || edge_idx >= NUM_EDGES) return;
    if (is_edge_occupied(edge_idx)) return;

    // Place the piece
    edge_pieces[edge_idx].type = PieceType::Road;
    edge_pieces[edge_idx].owner = player_idx;

    // Update player state
    PlayerState& player = players[player_idx];
    if (player.roads_remaining > 0) {
        player.roads_remaining--;
    }

    // Roads affect longest road, which affects VP
    // For now, caller should call update_longest_road() after placing roads
}

std::uint8_t GameState::roll_dice(std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> die(1, 6);
    
    std::uint8_t roll = static_cast<std::uint8_t>(die(rng) + die(rng));
    last_dice_roll = roll;
    
    // Distribute resources if not a 7
    if (roll != 7) {
        distribute_resources(roll);
    }
    
    return roll;
}

void GameState::distribute_resources(std::uint8_t dice_value) {
    if (dice_value == 7 || dice_value < 2 || dice_value > 12) return;

    // For each tile with the rolled number
    for (std::uint8_t t = 0; t < NUM_TILES; ++t) {
        const Tile& tile = board.tiles[t];
        
        // Skip if robber is on this tile
        if (t == robber_tile) continue;
        
        // Skip if tile doesn't match rolled number
        if (tile.number != dice_value) continue;
        
        // Skip desert tiles
        if (tile.resource == Resource::Desert) continue;
        
        // Get resource type for this tile
        ResourceType res_type = tile_resource_to_player_resource(tile.resource);
        if (res_type == ResourceType::COUNT) continue;
        
        // Get vertices of this tile
        std::uint8_t vertices[6];
        board.get_tile_vertices(t, vertices);
        
        // Give resources to players with settlements/cities on these vertices
        for (std::uint8_t v_idx : vertices) {
            const Piece& piece = vertex_pieces[v_idx];
            if (piece.type == PieceType::None) continue;
            if (piece.owner >= num_players) continue;
            
            PlayerState& player = players[piece.owner];
            std::uint8_t amount = (piece.type == PieceType::Settlement) ? 1 : 2;
            
            // Check resource bank limit before distributing
            std::size_t res_idx = static_cast<std::size_t>(res_type);
            std::uint8_t available = resource_bank[res_idx];
            std::uint8_t to_give = std::min(amount, available);
            
            player.resources[res_idx] += to_give;
            resource_bank[res_idx] -= to_give;
        }
    }
}

void GameState::move_robber(std::uint8_t tile_idx) {
    if (tile_idx >= NUM_TILES) return;
    robber_tile = tile_idx;
}

DevCardType GameState::buy_dev_card(std::uint8_t player_idx, std::uint32_t seed) {
    if (player_idx >= num_players) return DevCardType::COUNT;

    // Count total cards remaining
    std::uint32_t total_cards = 0;
    for (std::uint8_t count : dev_deck) {
        total_cards += count;
    }
    if (total_cards == 0) return DevCardType::COUNT;

    // Pick a random card from remaining deck
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::uint32_t> dist(0, total_cards - 1);
    std::uint32_t card_idx = dist(rng);

    // Find which card type this index corresponds to
    DevCardType card_type = DevCardType::COUNT;
    std::uint32_t cumulative = 0;
    for (std::size_t i = 0; i < NUM_DEV_CARD_TYPES; ++i) {
        cumulative += dev_deck[i];
        if (card_idx < cumulative) {
            card_type = static_cast<DevCardType>(i);
            dev_deck[i]--;
            break;
        }
    }

    if (card_type == DevCardType::COUNT) return DevCardType::COUNT;

    // Add to player's hand (bought this turn, cannot play until next turn)
    PlayerState& player = players[player_idx];
    player.dev_cards_bought_this_turn[static_cast<std::size_t>(card_type)]++;

    // Victory point cards immediately add hidden VP
    if (card_type == DevCardType::VictoryPoint) {
        player.hidden_victory_points++;
    }

    return card_type;
}

void GameState::play_knight(std::uint8_t player_idx) {
    if (player_idx >= num_players) return;

    PlayerState& player = players[player_idx];
    std::size_t knight_idx = static_cast<std::size_t>(DevCardType::Knight);

    if (player.dev_cards[knight_idx] == 0) return;

    // Remove card from hand
    player.dev_cards[knight_idx]--;

    // Increment knights played
    player.knights_played++;

    // Update largest army
    update_largest_army();
    update_victory_points(player_idx);
}

void GameState::end_turn() {
    // Move newly bought dev cards to regular hand
    PlayerState& player = players[current_player];
    for (std::size_t i = 0; i < NUM_DEV_CARD_TYPES; ++i) {
        player.dev_cards[i] += player.dev_cards_bought_this_turn[i];
        player.dev_cards_bought_this_turn[i] = 0;
    }

    // Advance to next player
    current_player = (current_player + 1) % num_players;
    
    // Increment turn counter when we wrap back to player 0
    if (current_player == 0) {
        turn_number++;
    }

    // Reset turn phase
    turn_phase = TurnPhase::RollDice;
}

// Helper: DFS to find longest road path from a given edge for a player
// Returns the length of the longest path starting from this edge
static std::uint8_t dfs_longest_road(
    const GameState& state,
    std::uint8_t player_idx,
    std::uint8_t edge_idx,
    std::array<bool, NUM_EDGES>& visited
) {
    // Base case: already visited this edge
    if (visited[edge_idx]) return 0;

    // Mark this edge as visited
    visited[edge_idx] = true;

    std::uint8_t max_length = 0;

    // Explore both vertices of this edge
    const Edge& edge = state.board.edges[edge_idx];
    for (std::uint8_t v_idx : edge.vertices) {
        if (v_idx == INVALID_VERTEX) continue;

        // Check if this vertex blocks the road (opponent's settlement/city)
        const Piece& vertex_piece = state.vertex_pieces[v_idx];
        if (vertex_piece.type == PieceType::Settlement || vertex_piece.type == PieceType::City) {
            if (vertex_piece.owner != player_idx) {
                continue; // Opponent's building blocks the road
            }
        }

        // Explore all edges connected to this vertex
        const Vertex& vertex = state.board.vertices[v_idx];
        for (std::uint8_t next_edge_idx : vertex.edges) {
            if (next_edge_idx == INVALID_EDGE || next_edge_idx == edge_idx) continue;
            
            // Only follow roads owned by this player
            const Piece& next_edge_piece = state.edge_pieces[next_edge_idx];
            if (next_edge_piece.type != PieceType::Road || next_edge_piece.owner != player_idx) {
                continue;
            }

            // Recursively explore this path
            std::uint8_t path_length = dfs_longest_road(state, player_idx, next_edge_idx, visited);
            max_length = std::max(max_length, path_length);
        }
    }

    // Unmark for other paths (backtracking)
    visited[edge_idx] = false;

    return max_length + 1;
}

void GameState::update_longest_road() {
    std::uint8_t new_owner = 0xFF;
    std::uint8_t max_road_length = longest_road_length;

    // For each player, find their longest road
    for (std::uint8_t p = 0; p < num_players; ++p) {
        std::uint8_t player_max = 0;

        // Try starting from each of the player's roads
        for (std::uint8_t e = 0; e < NUM_EDGES; ++e) {
            const Piece& edge_piece = edge_pieces[e];
            if (edge_piece.type != PieceType::Road || edge_piece.owner != p) {
                continue;
            }

            // Start DFS from this edge
            std::array<bool, NUM_EDGES> visited{};
            std::uint8_t length = dfs_longest_road(*this, p, e, visited);
            player_max = std::max(player_max, length);
        }

        // Check if this player has the longest road (must be at least 5)
        if (player_max >= 5 && player_max > max_road_length) {
            max_road_length = player_max;
            new_owner = p;
        }
    }

    // Update ownership if changed
    if (new_owner != longest_road_owner) {
        // Remove from old owner
        if (longest_road_owner != 0xFF && longest_road_owner < num_players) {
            players[longest_road_owner].has_longest_road = false;
            update_victory_points(longest_road_owner);
        }
        
        // Give to new owner
        if (new_owner != 0xFF) {
            longest_road_owner = new_owner;
            longest_road_length = max_road_length;
            players[new_owner].has_longest_road = true;
            update_victory_points(new_owner);
        }
    }
}

void GameState::update_largest_army() {
    std::uint8_t new_owner = 0xFF;
    std::uint8_t max_knights = largest_army_count;

    // Find player with most knights (must be at least 3)
    for (std::uint8_t p = 0; p < num_players; ++p) {
        if (players[p].knights_played > max_knights) {
            max_knights = players[p].knights_played;
            new_owner = p;
        }
    }

    // Update if changed
    if (new_owner != largest_army_owner && new_owner != 0xFF) {
        // Remove from old owner
        if (largest_army_owner != 0xFF && largest_army_owner < num_players) {
            players[largest_army_owner].has_largest_army = false;
            update_victory_points(largest_army_owner);
        }
        
        // Give to new owner
        largest_army_owner = new_owner;
        largest_army_count = max_knights;
        players[new_owner].has_largest_army = true;
        update_victory_points(new_owner);
    }
}

void GameState::update_victory_points(std::uint8_t player_idx) {
    if (player_idx >= num_players) return;

    PlayerState& player = players[player_idx];

    // Count public VP
    std::uint8_t public_vp = 0;

    // Settlements: 1 VP each
    // Cities: 2 VP each
    for (const Piece& piece : vertex_pieces) {
        if (piece.owner == player_idx) {
            if (piece.type == PieceType::Settlement) public_vp += 1;
            else if (piece.type == PieceType::City) public_vp += 2;
        }
    }

    // Longest road: 2 VP
    if (player.has_longest_road) public_vp += 2;

    // Largest army: 2 VP
    if (player.has_largest_army) public_vp += 2;

    player.public_victory_points = public_vp;

    // Count hidden VP from victory point dev cards
    std::size_t vp_idx = static_cast<std::size_t>(DevCardType::VictoryPoint);
    player.hidden_victory_points = player.dev_cards[vp_idx];
}

bool GameState::has_harbor_access(std::uint8_t player_idx, HarborType harbor_type) const {
    if (player_idx >= num_players) return false;
    
    // Check each harbor
    for (const Harbor& harbor : board.harbors) {
        if (harbor.type != harbor_type) continue;
        
        // Check if player has settlement/city at either endpoint of the harbor edge
        const Edge& edge = board.edges[harbor.edge_idx];
        for (std::uint8_t vertex_idx : edge.vertices) {
            if (vertex_idx == INVALID_VERTEX) continue;
            
            const Piece& piece = vertex_pieces[vertex_idx];
            if (piece.owner == player_idx && 
                (piece.type == PieceType::Settlement || piece.type == PieceType::City)) {
                return true;
            }
        }
    }
    
    return false;
}

std::uint8_t GameState::get_trade_ratio(std::uint8_t player_idx, ResourceType resource) const {
    if (player_idx >= num_players || resource == ResourceType::COUNT) {
        return 4; // Default bank trade
    }
    
    // Check for specific 2:1 harbor
    HarborType specific_harbor;
    switch (resource) {
        case ResourceType::Brick:  specific_harbor = HarborType::Brick;  break;
        case ResourceType::Lumber: specific_harbor = HarborType::Lumber; break;
        case ResourceType::Wool:   specific_harbor = HarborType::Wool;   break;
        case ResourceType::Grain:  specific_harbor = HarborType::Grain;  break;
        case ResourceType::Ore:    specific_harbor = HarborType::Ore;    break;
        default: specific_harbor = HarborType::None; break;
    }
    
    if (specific_harbor != HarborType::None && has_harbor_access(player_idx, specific_harbor)) {
        return 2; // 2:1 specific harbor
    }
    
    // Check for generic 3:1 harbor
    if (has_harbor_access(player_idx, HarborType::Generic)) {
        return 3; // 3:1 generic harbor
    }
    
    return 4; // Default bank trade
}

} // namespace catan
