// state_encoder.cpp
// Fast C++ implementation of state encoder for Catan RL.
// Optimized for minimal allocations and maximum speed.

#include "state_encoder.h"
#include <algorithm>
#include <cstring>

namespace catan {

std::vector<float> StateEncoder::encode_state(
    const GameState& state,
    std::uint8_t perspective_player
) const {
    // Allocate output buffer once
    std::vector<float> features(StateEncoderConfig::TOTAL_FEATURES);
    float* ptr = features.data();
    
    // Encode each category in sequence (writing directly to buffer)
    encode_tiles(state, ptr);
    ptr += StateEncoderConfig::NUM_TILES * StateEncoderConfig::TILE_FEATURES_PER;
    
    encode_vertices(state, perspective_player, ptr);
    ptr += StateEncoderConfig::NUM_VERTICES * StateEncoderConfig::VERTEX_FEATURES_PER;
    
    encode_edges(state, perspective_player, ptr);
    ptr += StateEncoderConfig::NUM_EDGES * StateEncoderConfig::EDGE_FEATURES_PER;
    
    encode_players(state, perspective_player, ptr);
    ptr += StateEncoderConfig::MAX_PLAYERS * StateEncoderConfig::PLAYER_FEATURES_PER;
    
    encode_global(state, perspective_player, ptr);
    
    return features;
}

void StateEncoder::encode_tiles(
    const GameState& state,
    float* output
) const {
    float* ptr = output;
    
    for (std::size_t tile_idx = 0; tile_idx < StateEncoderConfig::NUM_TILES; ++tile_idx) {
        const Tile& tile = state.board.tiles[tile_idx];
        
        // One-hot encode resource type (6 types: Desert, Brick, Lumber, Wool, Grain, Ore)
        std::uint8_t resource_val = static_cast<std::uint8_t>(tile.resource);
        for (std::size_t i = 0; i < 6; ++i) {
            ptr[i] = (i == resource_val) ? 1.0f : 0.0f;
        }
        ptr += 6;
        
        // Normalized dice number (0-1 range)
        *ptr++ = (tile.number > 0) ? (tile.number / 12.0f) : 0.0f;
        
        // Robber on this tile
        *ptr++ = (state.robber_tile == tile_idx) ? 1.0f : 0.0f;
    }
}

void StateEncoder::encode_vertices(
    const GameState& state,
    std::uint8_t perspective_player,
    float* output
) const {
    float* ptr = output;
    
    for (std::size_t vertex_idx = 0; vertex_idx < StateEncoderConfig::NUM_VERTICES; ++vertex_idx) {
        const Piece& piece = state.vertex_pieces[vertex_idx];
        
        // Piece type: empty, settlement, city (one-hot)
        ptr[0] = (piece.type == PieceType::None) ? 1.0f : 0.0f;
        ptr[1] = (piece.type == PieceType::Settlement) ? 1.0f : 0.0f;
        ptr[2] = (piece.type == PieceType::City) ? 1.0f : 0.0f;
        ptr += 3;
        
        // Owner relative to perspective (one-hot: none, self, opponent1, opponent2, opponent3)
        std::memset(ptr, 0, 5 * sizeof(float));
        if (piece.type == PieceType::None) {
            ptr[0] = 1.0f;
        } else {
            std::uint8_t owner = piece.owner;
            if (owner == perspective_player) {
                ptr[1] = 1.0f;  // Self
            } else if (owner < state.num_players) {
                // Opponents (relative indexing)
                std::uint8_t opponent_idx = (owner - perspective_player) % state.num_players;
                if (opponent_idx >= 1 && opponent_idx <= 3) {
                    ptr[1 + opponent_idx] = 1.0f;
                }
            }
        }
        ptr += 5;
        
        // Harbor access
        *ptr++ = get_harbor_at_vertex(state, static_cast<std::uint8_t>(vertex_idx));
    }
}

void StateEncoder::encode_edges(
    const GameState& state,
    std::uint8_t perspective_player,
    float* output
) const {
    float* ptr = output;
    
    for (std::size_t edge_idx = 0; edge_idx < StateEncoderConfig::NUM_EDGES; ++edge_idx) {
        const Piece& piece = state.edge_pieces[edge_idx];
        
        // Road presence (binary)
        *ptr++ = (piece.type == PieceType::Road) ? 1.0f : 0.0f;
        
        // Owner relative to perspective (one-hot: self, opponent1, opponent2, opponent3)
        std::memset(ptr, 0, 4 * sizeof(float));
        if (piece.type == PieceType::Road) {
            std::uint8_t owner = piece.owner;
            if (owner == perspective_player) {
                ptr[0] = 1.0f;  // Self
            } else if (owner < state.num_players) {
                std::uint8_t opponent_idx = (owner - perspective_player) % state.num_players;
                if (opponent_idx >= 1 && opponent_idx <= 3) {
                    ptr[opponent_idx] = 1.0f;
                }
            }
        }
        ptr += 4;
    }
}

void StateEncoder::encode_players(
    const GameState& state,
    std::uint8_t perspective_player,
    float* output
) const {
    float* ptr = output;
    
    for (std::size_t offset = 0; offset < StateEncoderConfig::MAX_PLAYERS; ++offset) {
        std::uint8_t player_idx = (perspective_player + offset) % state.num_players;
        
        if (player_idx < state.num_players) {
            const PlayerState& player = state.players[player_idx];
            
            // Resources (normalized by max reasonable amount ~20)
            for (std::size_t r = 0; r < StateEncoderConfig::NUM_RESOURCE_TYPES; ++r) {
                *ptr++ = player.resources[r] / 20.0f;
            }
            
            // Dev cards (normalized by max in deck)
            for (std::size_t d = 0; d < StateEncoderConfig::NUM_DEV_CARD_TYPES; ++d) {
                *ptr++ = player.dev_cards[d] / 5.0f;
            }
            
            // Pieces remaining (normalized by starting amount)
            *ptr++ = player.settlements_remaining / 5.0f;
            *ptr++ = player.cities_remaining / 4.0f;
            *ptr++ = player.roads_remaining / 15.0f;
            
            // Victory points (normalized by win condition)
            *ptr++ = player.public_victory_points / 10.0f;
            *ptr++ = player.hidden_victory_points / 10.0f;
            
            // Knights played (for largest army)
            *ptr++ = player.knights_played / 10.0f;
            
            // Special achievements
            *ptr++ = player.has_longest_road ? 1.0f : 0.0f;
            *ptr++ = player.has_largest_army ? 1.0f : 0.0f;
        } else {
            // Padding for games with <4 players
            std::memset(ptr, 0, StateEncoderConfig::PLAYER_FEATURES_PER * sizeof(float));
            ptr += StateEncoderConfig::PLAYER_FEATURES_PER;
        }
    }
}

void StateEncoder::encode_global(
    const GameState& state,
    std::uint8_t perspective_player,
    float* output
) const {
    float* ptr = output;
    
    // Current player (relative to perspective)
    std::uint8_t current_relative = (state.current_player - perspective_player) % state.num_players;
    for (std::size_t i = 0; i < 4; ++i) {
        ptr[i] = (i == current_relative) ? 1.0f : 0.0f;
    }
    ptr += 4;
    
    // Game phase (one-hot: Setup, MainGame)
    *ptr++ = (state.game_phase == GamePhase::Setup) ? 1.0f : 0.0f;
    *ptr++ = (state.game_phase == GamePhase::MainGame) ? 1.0f : 0.0f;
    
    // Turn phase (one-hot: RollDice, Discard, Robber, Trading)
    ptr[0] = (state.turn_phase == TurnPhase::RollDice) ? 1.0f : 0.0f;
    ptr[1] = (state.turn_phase == TurnPhase::Discard) ? 1.0f : 0.0f;
    ptr[2] = (state.turn_phase == TurnPhase::Robber) ? 1.0f : 0.0f;
    ptr[3] = (state.turn_phase == TurnPhase::Trading) ? 1.0f : 0.0f;
    ptr += 4;
    
    // Turn number (normalized)
    *ptr++ = state.turn_number / 100.0f;
    
    // Longest road owner (relative to perspective)
    if (state.longest_road_owner == 0xFF) {
        ptr[0] = 1.0f;
        ptr[1] = ptr[2] = ptr[3] = 0.0f;
    } else {
        std::uint8_t lr_relative = (state.longest_road_owner - perspective_player) % state.num_players;
        for (std::size_t i = 0; i < 4; ++i) {
            ptr[i] = (i == lr_relative) ? 1.0f : 0.0f;
        }
    }
    ptr += 4;
    
    // Largest army owner (relative to perspective)
    if (state.largest_army_owner == 0xFF) {
        ptr[0] = 1.0f;
        ptr[1] = ptr[2] = ptr[3] = 0.0f;
    } else {
        std::uint8_t la_relative = (state.largest_army_owner - perspective_player) % state.num_players;
        for (std::size_t i = 0; i < 4; ++i) {
            ptr[i] = (i == la_relative) ? 1.0f : 0.0f;
        }
    }
    ptr += 4;
    
    // Number of players in game
    *ptr++ = state.num_players / 4.0f;
}

float StateEncoder::get_harbor_at_vertex(
    const GameState& state,
    std::uint8_t vertex_idx
) const {
    const Vertex& vertex = state.board.vertices[vertex_idx];
    
    for (std::size_t i = 0; i < vertex.edges.size(); ++i) {
        std::uint8_t edge_idx = vertex.edges[i];
        if (edge_idx == INVALID_EDGE) continue;
        
        // Check all harbors
        for (std::size_t h = 0; h < state.board.harbors.size(); ++h) {
            const Harbor& harbor = state.board.harbors[h];
            if (harbor.edge_idx == edge_idx) {
                // Harbor found
                if (harbor.type == HarborType::Generic) {
                    return 0.5f;  // 3:1 harbor
                } else {
                    return 1.0f;  // 2:1 specific harbor
                }
            }
        }
    }
    
    return 0.0f;  // No harbor
}

} // namespace catan
