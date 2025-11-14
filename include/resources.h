// resources.h
// Resource types for player hands and trading.
// Separate from board tile resources for clarity.

#pragma once

#include <cstdint>
#include "board_grid.h"

namespace catan {

// Resource types that players collect and trade.
// Standard Catan has 5 resource types.
enum class ResourceType : std::uint8_t {
    Brick = 0,
    Lumber,
    Wool,
    Grain,
    Ore,
    COUNT  // Sentinel for array sizing
};

constexpr std::size_t NUM_RESOURCE_TYPES = 5;

// Convert board tile resource to player resource type.
// Desert tiles produce no resources.
inline ResourceType tile_resource_to_player_resource(Resource tile_res) {
    switch (tile_res) {
        case Resource::Brick:  return ResourceType::Brick;
        case Resource::Lumber: return ResourceType::Lumber;
        case Resource::Wool:   return ResourceType::Wool;
        case Resource::Grain:  return ResourceType::Grain;
        case Resource::Ore:    return ResourceType::Ore;
        default:               return ResourceType::COUNT; // Invalid/Desert
    }
}

} // namespace catan
