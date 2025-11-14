// dev_cards.h
// Development card types and distribution for Catan.

#pragma once

#include <cstdint>
#include <array>

namespace catan {

// Development card types in standard Catan.
enum class DevCardType : std::uint8_t {
    Knight = 0,      // 14 knights in deck
    VictoryPoint,    // 5 victory point cards
    RoadBuilding,    // 2 road building cards
    YearOfPlenty,    // 2 year of plenty cards
    Monopoly,        // 2 monopoly cards
    COUNT            // Sentinel for array sizing
};

constexpr std::size_t NUM_DEV_CARD_TYPES = 5;

// Standard Catan development card distribution (25 total cards).
constexpr std::array<std::uint8_t, NUM_DEV_CARD_TYPES> STANDARD_DEV_CARD_COUNTS = {
    14,  // Knight
    5,   // VictoryPoint
    2,   // RoadBuilding
    2,   // YearOfPlenty
    2    // Monopoly
};

constexpr std::size_t TOTAL_DEV_CARDS = 25;

} // namespace catan
