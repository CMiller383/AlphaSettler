// board_grid.h
// Minimal, fast Catan board grid representation for RL engine.
// - Plain structs, enums, and fixed-size arrays
// - No dynamic allocation in core representation

#pragma once

#include <array>
#include <cstdint>

namespace catan {

// Catan base board uses 19 hex tiles arranged in a fixed layout.
constexpr std::size_t NUM_TILES = 19;

// Standard Catan board has 54 intersections (vertices) and 72 edges.
// These are the only locations where pieces can be placed, so we
// pre-index them and use fixed-size arrays for fast access.
constexpr std::size_t NUM_VERTICES = 54;
constexpr std::size_t NUM_EDGES    = 72;

// Standard Catan has 9 harbors (ports) on the coast
constexpr std::size_t NUM_HARBORS = 9;

// Resource types for tiles (desert = none)
enum class Resource : std::uint8_t {
    Desert = 0,
    Brick,
    Lumber,
    Wool,
    Grain,
    Ore,
};

// Harbor types for trading
enum class HarborType : std::uint8_t {
    None = 0,       // No harbor
    Generic,        // 3:1 any resource
    Brick,          // 2:1 brick
    Lumber,         // 2:1 lumber
    Wool,           // 2:1 wool
    Grain,          // 2:1 grain
    Ore,            // 2:1 ore
};

// Simple axial-like index for tiles: we keep a fixed indexing 0..18.
// Adjacency is precomputed and stored as indices into this array.
// -1 (UINT8_MAX) is used for "no neighbor".
constexpr std::uint8_t INVALID_TILE   = 0xFFu;
constexpr std::uint8_t INVALID_VERTEX = 0xFFu;
constexpr std::uint8_t INVALID_EDGE   = 0xFFu;

struct Tile {
    Resource resource;     // Type of resource on this tile
    std::uint8_t number;   // Dice number 2..12 (0 for desert / none)
};

// Each vertex is the intersection of up to 3 tiles and up to 3 edges.
// We keep everything as small indices into fixed arrays.
struct Vertex {
    std::array<std::uint8_t, 3> tiles;   // up to 3 adjacent tiles (INVALID_TILE if none)
    std::array<std::uint8_t, 3> edges;   // up to 3 incident edges (INVALID_EDGE if none)
};

// Each edge connects 2 vertices and lies between up to 2 tiles.
struct Edge {
    std::array<std::uint8_t, 2> vertices; // endpoints (INVALID_VERTEX if none)
    std::array<std::uint8_t, 2> tiles;    // adjacent tiles (INVALID_TILE if none)
};

// Harbor location: defined by an edge on the coast and the harbor type
struct Harbor {
    std::uint8_t edge_idx;   // Edge where harbor is located
    HarborType type;         // Type of harbor (generic or specific resource)
};

struct BoardGrid {
    // Fixed list of tiles; layout is implicit in neighbor table.
    std::array<Tile, NUM_TILES> tiles{};

    // For each tile, up to 6 neighbors (clockwise). INVALID_TILE when absent.
    // This is static for the standard board and can be shared.
    std::array<std::array<std::uint8_t, 6>, NUM_TILES> neighbors{};

    // Fixed topology for vertices and edges on the standard board.
    // These are static in practice but stored here for simplicity and
    // to keep the representation self-contained.
    std::array<Vertex, NUM_VERTICES> vertices{};
    std::array<Edge,   NUM_EDGES>    edges{};
    
    // Harbors: 9 total (4 generic 3:1, 5 specific 2:1)
    std::array<Harbor, NUM_HARBORS> harbors{};

    // Initialize tiles and adjacency to a standard layout.
    // Caller can then randomize resources/numbers if desired.
    static BoardGrid create_default();

    // Randomize resources and dice numbers based on seed.
    // Uses standard Catan distribution: 4 lumber, 4 grain, 4 wool, 3 brick, 3 ore, 1 desert.
    // Numbers: 2-12 excluding 7, with proper frequencies.
    void randomize(std::uint32_t seed);

    // Fast lookup: Get all vertices adjacent to a given tile.
    // Returns count of valid vertices (max 6).
    std::uint8_t get_tile_vertices(std::uint8_t tile_idx, std::uint8_t out_vertices[6]) const;

    // Fast lookup: Get all edges adjacent to a given tile.
    // Returns count of valid edges (max 6).
    std::uint8_t get_tile_edges(std::uint8_t tile_idx, std::uint8_t out_edges[6]) const;
};

} // namespace catan
