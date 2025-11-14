// test_visualize_grid.cpp
// Test harness to construct, randomize, and visualize the Catan board grid.
// Demonstrates complete vertex/edge topology and seed-based randomization.

#include <iostream>
#include <iomanip>
#include <array>
#include <string>

#include "board_grid.h"

using namespace catan;

static char resource_to_char(Resource r) {
    switch (r) {
        case Resource::Desert: return 'D';
        case Resource::Brick:  return 'B';
        case Resource::Lumber: return 'L';
        case Resource::Wool:   return 'W';
        case Resource::Grain:  return 'G';
        case Resource::Ore:    return 'O';
        default:               return '?';
    }
}

static const char* resource_to_string(Resource r) {
    switch (r) {
        case Resource::Desert: return "Desert";
        case Resource::Brick:  return "Brick ";
        case Resource::Lumber: return "Lumber";
        case Resource::Wool:   return "Wool  ";
        case Resource::Grain:  return "Grain ";
        case Resource::Ore:    return "Ore   ";
        default:               return "???   ";
    }
}

static void print_board_layout(const BoardGrid& board) {
    std::cout << "\n=== Catan Board Layout (3-4-5-4-3) ===\n\n";

    auto print_row = [&](const std::initializer_list<int>& indices, int indent) {
        for (int i = 0; i < indent; ++i) std::cout << ' ';
        for (int idx : indices) {
            if (idx < 0) {
                std::cout << "      ";
            } else {
                const auto& tile = board.tiles[idx];
                char res = resource_to_char(tile.resource);
                int num = tile.number;
                if (num > 0) {
                    std::cout << res << std::setw(2) << num << "  ";
                } else {
                    std::cout << res << "    ";
                }
            }
        }
        std::cout << '\n';
    };

    print_row({0, 1, 2}, 10);
    print_row({3, 4, 5, 6}, 6);
    print_row({7, 8, 9, 10, 11}, 2);
    print_row({12, 13, 14, 15}, 6);
    print_row({16, 17, 18}, 10);
}

static void print_tile_details(const BoardGrid& board) {
    std::cout << "\n=== Tile Details ===\n";
    std::cout << "Tile | Resource | Number\n";
    std::cout << "-----|----------|-------\n";
    for (std::size_t i = 0; i < NUM_TILES; ++i) {
        const auto& tile = board.tiles[i];
        std::cout << std::setw(4) << i << " | "
                  << resource_to_string(tile.resource) << " | "
                  << (tile.number > 0 ? std::to_string(tile.number) : "-")
                  << '\n';
    }
}

static void print_topology_sample(const BoardGrid& board) {
    std::cout << "\n=== Topology Sample (Tile 0) ===\n";
    
    // Get vertices and edges for tile 0
    std::uint8_t vertices[6];
    std::uint8_t edges[6];
    
    std::uint8_t v_count = board.get_tile_vertices(0, vertices);
    std::uint8_t e_count = board.get_tile_edges(0, edges);
    
    std::cout << "Tile 0 has " << static_cast<int>(v_count) << " vertices: ";
    for (std::uint8_t i = 0; i < v_count; ++i) {
        std::cout << static_cast<int>(vertices[i]) << " ";
    }
    std::cout << "\n";
    
    std::cout << "Tile 0 has " << static_cast<int>(e_count) << " edges: ";
    for (std::uint8_t i = 0; i < e_count; ++i) {
        std::cout << static_cast<int>(edges[i]) << " ";
    }
    std::cout << "\n";
    
    // Show vertex details
    if (v_count > 0) {
        std::uint8_t v = vertices[0];
        const Vertex& vertex = board.vertices[v];
        std::cout << "\nVertex " << static_cast<int>(v) << " details:\n";
        std::cout << "  Adjacent tiles: ";
        for (std::uint8_t t : vertex.tiles) {
            if (t != INVALID_TILE) std::cout << static_cast<int>(t) << " ";
        }
        std::cout << "\n  Adjacent edges: ";
        for (std::uint8_t e : vertex.edges) {
            if (e != INVALID_EDGE) std::cout << static_cast<int>(e) << " ";
        }
        std::cout << "\n";
    }
    
    // Show edge details
    if (e_count > 0) {
        std::uint8_t e = edges[0];
        const Edge& edge = board.edges[e];
        std::cout << "\nEdge " << static_cast<int>(e) << " details:\n";
        std::cout << "  Connects vertices: ";
        for (std::uint8_t v : edge.vertices) {
            if (v != INVALID_VERTEX) std::cout << static_cast<int>(v) << " ";
        }
        std::cout << "\n  Adjacent tiles: ";
        for (std::uint8_t t : edge.tiles) {
            if (t != INVALID_TILE) std::cout << static_cast<int>(t) << " ";
        }
        std::cout << "\n";
    }
}

static void print_topology_counts(const BoardGrid& board) {
    std::cout << "\n=== Topology Summary ===\n";
    std::cout << "Total tiles:    " << NUM_TILES << '\n';
    std::cout << "Total vertices: " << NUM_VERTICES << '\n';
    std::cout << "Total edges:    " << NUM_EDGES << '\n';
    
    // Validate that each tile has 6 vertices and 6 edges
    std::cout << "\nValidating tile topology...\n";
    bool all_valid = true;
    for (std::uint8_t t = 0; t < NUM_TILES; ++t) {
        std::uint8_t vertices[6];
        std::uint8_t edges[6];
        std::uint8_t v_count = board.get_tile_vertices(t, vertices);
        std::uint8_t e_count = board.get_tile_edges(t, edges);
        
        if (v_count != 6 || e_count != 6) {
            std::cout << "  Tile " << static_cast<int>(t) 
                      << ": vertices=" << static_cast<int>(v_count)
                      << ", edges=" << static_cast<int>(e_count) << " (expected 6 each)\n";
            all_valid = false;
        }
    }
    if (all_valid) {
        std::cout << "  All tiles have 6 vertices and 6 edges - OK!\n";
    }
}

// Validate that for each tile, consecutive corner vertices are connected
// by exactly one edge that:
//  - is reported in that tile's edge list
//  - lists the tile among its adjacent tiles
//  - appears in both endpoint vertices' edge lists
static void validate_tile_edge_cycles(const BoardGrid& board) {
    std::cout << "\nValidating edge-vertex cycles per tile...\n";
    bool ok = true;

    for (std::uint8_t t = 0; t < NUM_TILES; ++t) {
        std::uint8_t tile_vertices[6];
        std::uint8_t tile_edges[6];
        board.get_tile_vertices(t, tile_vertices);
        board.get_tile_edges(t, tile_edges);

        for (int k = 0; k < 6; ++k) {
            std::uint8_t v  = tile_vertices[k];
            std::uint8_t vn = tile_vertices[(k + 1) % 6];

            bool found_edge = false;
            for (int i = 0; i < 6; ++i) {
                std::uint8_t e_idx = tile_edges[i];
                const Edge& e = board.edges[e_idx];
                std::uint8_t a = e.vertices[0];
                std::uint8_t b = e.vertices[1];

                if (!((a == v && b == vn) || (a == vn && b == v))) {
                    continue;
                }

                // Edge must list tile t among its tiles
                if (e.tiles[0] != t && e.tiles[1] != t) {
                    std::cout << "  ERROR: tile " << static_cast<int>(t)
                              << " edge " << static_cast<int>(e_idx)
                              << " connects vertices " << static_cast<int>(v)
                              << "," << static_cast<int>(vn)
                              << " but tiles are " << static_cast<int>(e.tiles[0])
                              << "," << static_cast<int>(e.tiles[1]) << "\n";
                    ok = false;
                }

                auto vertex_has_edge = [&](std::uint8_t v_idx) {
                    const Vertex& vert = board.vertices[v_idx];
                    for (std::uint8_t ve : vert.edges) {
                        if (ve == e_idx) return true;
                    }
                    return false;
                };

                if (!vertex_has_edge(v) || !vertex_has_edge(vn)) {
                    std::cout << "  ERROR: edge " << static_cast<int>(e_idx)
                              << " missing from vertex edge lists for vertices "
                              << static_cast<int>(v) << "," << static_cast<int>(vn) << "\n";
                    ok = false;
                }

                found_edge = true;
                break;
            }

            if (!found_edge) {
                std::cout << "  ERROR: tile " << static_cast<int>(t)
                          << " has no edge between vertices "
                          << static_cast<int>(v) << " and "
                          << static_cast<int>(vn) << "\n";
                ok = false;
            }
        }
    }

    if (ok) {
        std::cout << "  All tile edge-vertex cycles consistent - OK!\n";
    }
}

// Validate that for every vertex->tile relation, the reverse tile->vertex
// relation also holds.
static void validate_vertex_tile_adjacency(const BoardGrid& board) {
    std::cout << "\nValidating vertex->tile adjacency...\n";
    bool ok = true;

    for (std::uint8_t v_idx = 0; v_idx < NUM_VERTICES; ++v_idx) {
        const Vertex& vert = board.vertices[v_idx];
        for (std::uint8_t t : vert.tiles) {
            if (t == INVALID_TILE) continue;

            std::uint8_t tile_vertices[6];
            std::uint8_t count = board.get_tile_vertices(t, tile_vertices);
            bool found = false;
            for (std::uint8_t i = 0; i < count; ++i) {
                if (tile_vertices[i] == v_idx) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "  ERROR: vertex " << static_cast<int>(v_idx)
                          << " lists tile " << static_cast<int>(t)
                          << " but tile does not list vertex\n";
                ok = false;
            }
        }
    }

    if (ok) {
        std::cout << "  Vertex->tile adjacency consistent - OK!\n";
    }
}

// Validate that for every edge->tile relation, the reverse tile->edge
// relation also holds.
static void validate_edge_tile_adjacency(const BoardGrid& board) {
    std::cout << "\nValidating edge->tile adjacency...\n";
    bool ok = true;

    for (std::uint8_t e_idx = 0; e_idx < NUM_EDGES; ++e_idx) {
        const Edge& edge = board.edges[e_idx];
        for (std::uint8_t t : edge.tiles) {
            if (t == INVALID_TILE) continue;

            std::uint8_t tile_edges[6];
            std::uint8_t count = board.get_tile_edges(t, tile_edges);
            bool found = false;
            for (std::uint8_t i = 0; i < count; ++i) {
                if (tile_edges[i] == e_idx) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "  ERROR: edge " << static_cast<int>(e_idx)
                          << " lists tile " << static_cast<int>(t)
                          << " but tile does not list edge\n";
                ok = false;
            }
        }
    }

    if (ok) {
        std::cout << "  Edge->tile adjacency consistent - OK!\n";
    }
}

int main() {
    std::cout << "==================================\n";
    std::cout << "  Catan Board Grid Test\n";
    std::cout << "==================================\n";

    // Create default board
    BoardGrid board = BoardGrid::create_default();
    
    // Randomize with seed 42 for reproducibility
    std::cout << "\nRandomizing board with seed 42...\n";
    board.randomize(42);
    
    // Display board
    print_board_layout(board);
    print_tile_details(board);
    print_topology_sample(board);
    print_topology_counts(board);

    // Deep consistency checks for vertex/edge topology
    validate_tile_edge_cycles(board);
    validate_vertex_tile_adjacency(board);
    validate_edge_tile_adjacency(board);
    
    // Test determinism
    std::cout << "\n=== Testing Seed Determinism ===\n";
    BoardGrid board2 = BoardGrid::create_default();
    board2.randomize(42);
    
    bool same = true;
    for (std::size_t i = 0; i < NUM_TILES; ++i) {
        if (board.tiles[i].resource != board2.tiles[i].resource ||
            board.tiles[i].number != board2.tiles[i].number) {
            same = false;
            break;
        }
    }
    std::cout << "Two boards with seed 42: " << (same ? "IDENTICAL" : "DIFFERENT") << '\n';
    
    // Test different seed
    BoardGrid board3 = BoardGrid::create_default();
    board3.randomize(123);
    
    bool different = false;
    for (std::size_t i = 0; i < NUM_TILES; ++i) {
        if (board.tiles[i].resource != board3.tiles[i].resource ||
            board.tiles[i].number != board3.tiles[i].number) {
            different = true;
            break;
        }
    }
    std::cout << "Board with seed 42 vs 123: " << (different ? "DIFFERENT" : "IDENTICAL") << '\n';
    
    std::cout << "\n==================================\n";
    std::cout << "  Test Complete!\n";
    std::cout << "==================================\n";

    return 0;
}
