// board_grid.cpp
// Implementation of minimal Catan board grid optimized for RL self-play.

#include "board_grid.h"
#include <algorithm>
#include <random>

namespace catan {

// Standard Catan has 19 hexes. We'll use a fixed index layout and neighbor table.
// Index layout (row-wise, common convention):
//   Row 0:       0   1   2
//   Row 1:     3   4   5   6
//   Row 2:   7   8   9  10  11
//   Row 3:    12  13  14  15
//   Row 4:       16  17  18
//
// Neighbor table is hard-coded for speed and determinism.

static constexpr std::array<std::array<std::uint8_t, 6>, NUM_TILES> STANDARD_NEIGHBORS = { {
    // 0
    { 1, 4, 3, INVALID_TILE, INVALID_TILE, INVALID_TILE },
    // 1
    { 2, 5, 4, 0, INVALID_TILE, INVALID_TILE },
    // 2
    { INVALID_TILE, 6, 5, 1, INVALID_TILE, INVALID_TILE },
    // 3
    { 4, 8, 7, INVALID_TILE, INVALID_TILE, 0 },
    // 4
    { 5, 9, 8, 7, 3, 0 },
    // 5
    { 6, 10, 9, 8, 4, 1 },
    // 6
    { INVALID_TILE, 11, 10, 9, 5, 2 },
    // 7
    { 8, 13, 12, INVALID_TILE, INVALID_TILE, 3 },
    // 8
    { 9, 14, 13, 12, 7, 4 },
    // 9
    { 10, 15, 14, 13, 8, 5 },
    // 10
    { 11, INVALID_TILE, 15, 14, 9, 6 },
    // 11
    { INVALID_TILE, INVALID_TILE, INVALID_TILE, 15, 10, 6 },
    // 12
    { 13, 17, 16, INVALID_TILE, INVALID_TILE, 7 },
    // 13
    { 14, 18, 17, 16, 12, 8 },
    // 14
    { 15, INVALID_TILE, 18, 17, 13, 9 },
    // 15
    { INVALID_TILE, INVALID_TILE, INVALID_TILE, 18, 14, 10 },
    // 16
    { 17, INVALID_TILE, INVALID_TILE, INVALID_TILE, INVALID_TILE, 12 },
    // 17
    { 18, INVALID_TILE, INVALID_TILE, INVALID_TILE, 16, 13 },
    // 18
    { INVALID_TILE, INVALID_TILE, INVALID_TILE, INVALID_TILE, 17, 14 },
} };

// Canonical mapping from tiles to their 6 corner vertices and edges
// (clockwise order), derived from a reference open-source Catan
// implementation. All higher-level topology (vertices/edges arrays)
// is generated from these two tables to avoid manual inconsistencies.

static constexpr std::array<std::array<std::uint8_t, 6>, NUM_TILES> TILE_VERTICES = {{
    {{0, 1, 2, 10, 9, 8}},     // tile 0
    {{2, 3, 4, 12, 11, 10}},   // tile 1
    {{4, 5, 6, 14, 13, 12}},   // tile 2
    {{7, 8, 9, 19, 18, 17}},   // tile 3
    {{9, 10, 11, 21, 20, 19}}, // tile 4
    {{11, 12, 13, 23, 22, 21}},// tile 5
    {{13, 14, 15, 25, 24, 23}},// tile 6
    {{16, 17, 18, 29, 28, 27}},// tile 7
    {{18, 19, 20, 31, 30, 29}},// tile 8
    {{20, 21, 22, 33, 32, 31}},// tile 9
    {{22, 23, 24, 35, 34, 33}},// tile 10
    {{24, 25, 26, 37, 36, 35}},// tile 11
    {{28, 29, 30, 40, 39, 38}},// tile 12
    {{30, 31, 32, 42, 41, 40}},// tile 13
    {{32, 33, 34, 44, 43, 42}},// tile 14
    {{34, 35, 36, 46, 45, 44}},// tile 15
    {{39, 40, 41, 49, 48, 47}},// tile 16
    {{41, 42, 43, 51, 50, 49}},// tile 17
    {{43, 44, 45, 53, 52, 51}},// tile 18
}};

static constexpr std::array<std::array<std::uint8_t, 6>, NUM_TILES> TILE_EDGES = {{
    {{0, 1, 2, 3, 4, 5}},        // tile 0
    {{6, 7, 8, 9, 10, 2}},       // tile 1
    {{11, 12, 13, 14, 15, 8}},   // tile 2
    {{16, 4, 17, 18, 19, 20}},   // tile 3
    {{3, 10, 21, 22, 23, 17}},   // tile 4
    {{9, 15, 24, 25, 26, 21}},   // tile 5
    {{14, 27, 28, 29, 30, 24}},  // tile 6
    {{31, 19, 32, 33, 34, 35}},  // tile 7
    {{18, 23, 36, 37, 38, 32}},  // tile 8
    {{22, 26, 39, 40, 41, 36}},  // tile 9
    {{25, 30, 42, 43, 44, 39}},  // tile 10
    {{29, 45, 46, 47, 48, 42}},  // tile 11
    {{33, 38, 49, 50, 51, 52}},  // tile 12
    {{37, 41, 53, 54, 55, 49}},  // tile 13
    {{40, 44, 56, 57, 58, 53}},  // tile 14
    {{43, 48, 59, 60, 61, 56}},  // tile 15
    {{50, 55, 62, 63, 64, 65}},  // tile 16
    {{54, 58, 66, 67, 68, 62}},  // tile 17
    {{57, 61, 69, 70, 71, 66}},  // tile 18
}};

BoardGrid BoardGrid::create_default() {
    BoardGrid board;

    // Copy static neighbor table
    board.neighbors = STANDARD_NEIGHBORS;

    // Initialize vertices and edges to invalid
    for (auto& v : board.vertices) {
        v.tiles = { INVALID_TILE, INVALID_TILE, INVALID_TILE };
        v.edges = { INVALID_EDGE, INVALID_EDGE, INVALID_EDGE };
    }
    for (auto& e : board.edges) {
        e.vertices = { INVALID_VERTEX, INVALID_VERTEX };
        e.tiles    = { INVALID_TILE,   INVALID_TILE   };
    }

    // First pass: attach tiles to vertices and edges from TILE_* tables
    for (std::uint8_t t = 0; t < NUM_TILES; ++t) {
        for (std::uint8_t k = 0; k < 6; ++k) {
            const std::uint8_t v_idx = TILE_VERTICES[t][k];
            const std::uint8_t e_idx = TILE_EDGES[t][k];

            // Attach tile to vertex (up to 3 tiles per vertex)
            Vertex& v = board.vertices[v_idx];
            for (std::uint8_t& vt : v.tiles) {
                if (vt == INVALID_TILE) {
                    vt = t;
                    break;
                }
            }

            // Attach tile to edge (up to 2 tiles per edge)
            Edge& e = board.edges[e_idx];
            for (std::uint8_t& et : e.tiles) {
                if (et == INVALID_TILE) {
                    et = t;
                    break;
                }
            }

            // Define edge endpoints exactly as in reference implementation:
            // this edge is the side between vertex[k] and vertex[(k+1)%6] of tile t.
            const std::uint8_t v_next_idx = TILE_VERTICES[t][(k + 1) % 6];
            if (e.vertices[0] == INVALID_VERTEX && e.vertices[1] == INVALID_VERTEX) {
                e.vertices[0] = v_idx;
                e.vertices[1] = v_next_idx;
            } else {
                // If already set from another tile, ensure consistency
                // (in a valid board, both tiles will agree on endpoints).
                if (!((e.vertices[0] == v_idx && e.vertices[1] == v_next_idx) ||
                      (e.vertices[0] == v_next_idx && e.vertices[1] == v_idx))) {
                    // Inconsistent topology; leave as-is in release builds.
                }
            }
        }
    }

    // Derive incident edges per vertex from edge endpoints
    for (std::uint8_t e_idx = 0; e_idx < NUM_EDGES; ++e_idx) {
        const Edge& e = board.edges[e_idx];
        for (std::uint8_t endpoint = 0; endpoint < 2; ++endpoint) {
            const std::uint8_t v_idx = e.vertices[endpoint];
            if (v_idx == INVALID_VERTEX) continue;
            Vertex& v = board.vertices[v_idx];
            for (std::uint8_t& ve : v.edges) {
                if (ve == INVALID_EDGE) {
                    ve = e_idx;
                    break;
                }
            }
        }
    }

    // Initialize tiles with neutral defaults
    for (std::size_t i = 0; i < NUM_TILES; ++i) {
        board.tiles[i].resource = Resource::Desert;
        board.tiles[i].number = 0;
    }
    
    // Initialize harbors on coastal edges
    // Standard Catan has 9 harbors: 4 generic (3:1) and 5 specific (2:1)
    // These edge indices correspond to coastal edges around the board perimeter
    board.harbors[0] = { 0,  HarborType::Generic };  // Top-left coast
    board.harbors[1] = { 7,  HarborType::Lumber };   // Top-right coast
    board.harbors[2] = { 13, HarborType::Generic };  // Right coast (upper)
    board.harbors[3] = { 28, HarborType::Brick };    // Right coast (lower)
    board.harbors[4] = { 46, HarborType::Generic };  // Bottom-right coast
    board.harbors[5] = { 60, HarborType::Grain };    // Bottom-left coast
    board.harbors[6] = { 64, HarborType::Generic };  // Left coast (lower)
    board.harbors[7] = { 35, HarborType::Ore };      // Left coast (middle)
    board.harbors[8] = { 20, HarborType::Wool };     // Left coast (upper)

    return board;
}

void BoardGrid::randomize(std::uint32_t seed) {
    // Standard Catan resource distribution:
    // 4 lumber, 4 grain, 4 wool, 3 brick, 3 ore, 1 desert = 19 tiles
    std::array<Resource, NUM_TILES> resources = {
        Resource::Lumber, Resource::Lumber, Resource::Lumber, Resource::Lumber,
        Resource::Grain, Resource::Grain, Resource::Grain, Resource::Grain,
        Resource::Wool, Resource::Wool, Resource::Wool, Resource::Wool,
        Resource::Brick, Resource::Brick, Resource::Brick,
        Resource::Ore, Resource::Ore, Resource::Ore,
        Resource::Desert
    };

    // Standard Catan number distribution (excluding 7):
    // Two each of 3,4,5,6,8,9,10,11 and one each of 2,12 = 18 numbers
    std::array<std::uint8_t, 18> numbers = {
        2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12
    };

    // Shuffle with deterministic seed
    std::mt19937 rng(seed);
    std::shuffle(resources.begin(), resources.end(), rng);
    std::shuffle(numbers.begin(), numbers.end(), rng);

    // Assign to tiles (desert gets no number)
    std::uint8_t number_idx = 0;
    for (std::size_t i = 0; i < NUM_TILES; ++i) {
        tiles[i].resource = resources[i];
        if (resources[i] == Resource::Desert) {
            tiles[i].number = 0;
        } else {
            tiles[i].number = numbers[number_idx++];
        }
    }
}

std::uint8_t BoardGrid::get_tile_vertices(std::uint8_t tile_idx, std::uint8_t out_vertices[6]) const {
    if (tile_idx >= NUM_TILES) return 0;

    // Use static lookup table for canonical clockwise ordering
    const auto& tile_verts = TILE_VERTICES[tile_idx];
    for (std::uint8_t i = 0; i < 6; ++i) {
        out_vertices[i] = tile_verts[i];
    }
    return 6;
}

std::uint8_t BoardGrid::get_tile_edges(std::uint8_t tile_idx, std::uint8_t out_edges[6]) const {
    if (tile_idx >= NUM_TILES) return 0;

    // Use static lookup table for canonical clockwise ordering
    const auto& tile_edgs = TILE_EDGES[tile_idx];
    for (std::uint8_t i = 0; i < 6; ++i) {
        out_edges[i] = tile_edgs[i];
    }
    return 6;
}

} // namespace catan
