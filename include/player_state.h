// player_state.h
// Per-player state representation optimized for RL engine.
// Uses fixed-size arrays and plain structs for cache efficiency.

#pragma once

#include <cstdint>
#include <array>
#include "resources.h"
#include "dev_cards.h"

namespace catan {

// Standard Catan allows up to 4 players.
constexpr std::size_t MAX_PLAYERS = 4;

// Piece limits per player in standard Catan.
constexpr std::uint8_t MAX_SETTLEMENTS_PER_PLAYER = 5;
constexpr std::uint8_t MAX_CITIES_PER_PLAYER = 4;
constexpr std::uint8_t MAX_ROADS_PER_PLAYER = 15;

// Player state: resources, development cards, pieces, and scores.
// All fields are plain data types for fast copying and minimal cache footprint.
struct PlayerState {
    // Resource cards in hand (count per resource type).
    std::array<std::uint8_t, NUM_RESOURCE_TYPES> resources{};

    // Development cards in hand (count per card type).
    // Newly bought cards cannot be played until next turn, tracked separately.
    std::array<std::uint8_t, NUM_DEV_CARD_TYPES> dev_cards{};
    std::array<std::uint8_t, NUM_DEV_CARD_TYPES> dev_cards_bought_this_turn{};

    // Knights played (for largest army calculation).
    std::uint8_t knights_played{0};

    // Pieces remaining in supply (not yet placed on board).
    std::uint8_t settlements_remaining{MAX_SETTLEMENTS_PER_PLAYER};
    std::uint8_t cities_remaining{MAX_CITIES_PER_PLAYER};
    std::uint8_t roads_remaining{MAX_ROADS_PER_PLAYER};

    // Victory points (public and hidden).
    // Public VP: settlements, cities, longest road, largest army.
    // Hidden VP: victory point development cards (revealed when winning).
    std::uint8_t public_victory_points{0};
    std::uint8_t hidden_victory_points{0};

    // Flags for special achievements.
    bool has_longest_road{false};
    bool has_largest_army{false};

    // Get total resource count (for robber/trading checks).
    std::uint8_t total_resources() const {
        std::uint8_t total = 0;
        for (std::uint8_t count : resources) {
            total += count;
        }
        return total;
    }

    // Get total victory points (public + hidden).
    std::uint8_t total_victory_points() const {
        return public_victory_points + hidden_victory_points;
    }

    // Check if player has sufficient resources for a cost.
    bool can_afford(const std::array<std::uint8_t, NUM_RESOURCE_TYPES>& cost) const {
        for (std::size_t i = 0; i < NUM_RESOURCE_TYPES; ++i) {
            if (resources[i] < cost[i]) return false;
        }
        return true;
    }

    // Deduct resources (caller must check affordability first).
    void pay_resources(const std::array<std::uint8_t, NUM_RESOURCE_TYPES>& cost) {
        for (std::size_t i = 0; i < NUM_RESOURCE_TYPES; ++i) {
            resources[i] -= cost[i];
        }
    }

    // Add resources.
    void gain_resources(const std::array<std::uint8_t, NUM_RESOURCE_TYPES>& gain) {
        for (std::size_t i = 0; i < NUM_RESOURCE_TYPES; ++i) {
            resources[i] += gain[i];
        }
    }
};

// Standard costs for building pieces and buying development cards.
namespace BuildCost {
    constexpr std::array<std::uint8_t, NUM_RESOURCE_TYPES> ROAD = {1, 1, 0, 0, 0};        // 1 brick, 1 lumber
    constexpr std::array<std::uint8_t, NUM_RESOURCE_TYPES> SETTLEMENT = {1, 1, 1, 1, 0}; // 1 brick, 1 lumber, 1 wool, 1 grain
    constexpr std::array<std::uint8_t, NUM_RESOURCE_TYPES> CITY = {0, 0, 0, 2, 3};       // 2 grain, 3 ore
    constexpr std::array<std::uint8_t, NUM_RESOURCE_TYPES> DEV_CARD = {0, 0, 1, 1, 1};   // 1 wool, 1 grain, 1 ore
}

} // namespace catan
