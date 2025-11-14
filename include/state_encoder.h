// state_encoder.h
// Fast C++ state encoder for Catan RL neural network input.
// Converts GameState into ~900-dimensional feature vector.
// Optimized for minimal allocations and vectorized operations.

#pragma once

#include "game_state.h"
#include <vector>
#include <array>
#include <cstdint>

namespace catan {

// State encoder configuration
struct StateEncoderConfig {
    // Feature dimensions (must match Python encoder)
    static constexpr std::size_t NUM_TILES = 19;
    static constexpr std::size_t NUM_VERTICES = 54;
    static constexpr std::size_t NUM_EDGES = 72;
    static constexpr std::size_t NUM_HARBORS = 9;
    static constexpr std::size_t MAX_PLAYERS = 4;
    static constexpr std::size_t NUM_RESOURCE_TYPES = 5;
    static constexpr std::size_t NUM_DEV_CARD_TYPES = 5;
    
    // Feature counts per category
    static constexpr std::size_t TILE_FEATURES_PER = 8;      // resource(6) + number(1) + robber(1)
    static constexpr std::size_t VERTEX_FEATURES_PER = 9;    // piece_type(3) + owner(5) + harbor(1)
    static constexpr std::size_t EDGE_FEATURES_PER = 5;      // occupied(1) + owner(4)
    static constexpr std::size_t PLAYER_FEATURES_PER = 17;   // resources(5) + dev_cards(5) + pieces(3) + VP(2) + achievements(2)
    static constexpr std::size_t GLOBAL_FEATURES = 20;       // current_player(4) + phases(6) + turn(1) + achievements(8) + num_players(1)
    
    // Total feature count
    static constexpr std::size_t TOTAL_FEATURES = 
        (NUM_TILES * TILE_FEATURES_PER) +
        (NUM_VERTICES * VERTEX_FEATURES_PER) +
        (NUM_EDGES * EDGE_FEATURES_PER) +
        (MAX_PLAYERS * PLAYER_FEATURES_PER) +
        GLOBAL_FEATURES;
};

// Fast state encoder for neural network input
class StateEncoder {
public:
    StateEncoder() = default;
    
    // Encode game state from perspective player's view
    // Returns feature vector of size TOTAL_FEATURES (~900)
    std::vector<float> encode_state(
        const GameState& state,
        std::uint8_t perspective_player
    ) const;
    
    // Get total feature count
    static constexpr std::size_t get_feature_size() {
        return StateEncoderConfig::TOTAL_FEATURES;
    }
    
private:
    // Encoding helpers (write directly to output buffer for speed)
    void encode_tiles(
        const GameState& state,
        float* output
    ) const;
    
    void encode_vertices(
        const GameState& state,
        std::uint8_t perspective_player,
        float* output
    ) const;
    
    void encode_edges(
        const GameState& state,
        std::uint8_t perspective_player,
        float* output
    ) const;
    
    void encode_players(
        const GameState& state,
        std::uint8_t perspective_player,
        float* output
    ) const;
    
    void encode_global(
        const GameState& state,
        std::uint8_t perspective_player,
        float* output
    ) const;
    
    // Helper: check if vertex has harbor access
    float get_harbor_at_vertex(
        const GameState& state,
        std::uint8_t vertex_idx
    ) const;
};

} // namespace catan
