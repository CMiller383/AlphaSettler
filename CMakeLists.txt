cmake_minimum_required(VERSION 3.15)
project(CatanRLEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Core library
add_library(catan_core
    src/board_grid.cpp
    src/game_state.cpp
    src/move_gen.cpp
    src/state_transition.cpp
    src/state_encoder.cpp
)

target_include_directories(catan_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# MCTS library (pure MCTS with random rollouts)
add_library(catan_mcts
    src/mcts/mcts_search.cpp
    src/mcts/mcts_agent.cpp
)

target_link_libraries(catan_mcts PUBLIC catan_core)

target_include_directories(catan_mcts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# AlphaZero MCTS library (NN-guided MCTS)
add_library(catan_alphazero
    src/mcts/alphazero_mcts.cpp
)

target_link_libraries(catan_alphazero PUBLIC catan_core)

target_include_directories(catan_alphazero PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Simple visualization test executable
add_executable(test_visualize_grid
    tests/test_visualize_grid.cpp
)

target_link_libraries(test_visualize_grid PRIVATE catan_core)

# Game state test executable
add_executable(test_game_state
    tests/test_game_state.cpp
)

target_link_libraries(test_game_state PRIVATE catan_core)

# Move generation test executable
add_executable(test_move_gen
    tests/test_move_gen.cpp
)

target_link_libraries(test_move_gen PRIVATE catan_core)

# State transition test executable
add_executable(test_state_transition
    tests/test_state_transition.cpp
)

target_link_libraries(test_state_transition PRIVATE catan_core)

# MCTS test executable
add_executable(test_mcts
    tests/test_mcts.cpp
)

target_link_libraries(test_mcts PRIVATE catan_mcts)

# MCTS agent test executable
add_executable(test_mcts_agent
    tests/test_mcts_agent.cpp
)

target_link_libraries(test_mcts_agent PRIVATE catan_mcts)

# Optional: enable more warnings in development
if (MSVC)
    target_compile_options(catan_core PRIVATE /W4 /permissive-)
    target_compile_options(catan_mcts PRIVATE /W4 /permissive-)
    target_compile_options(test_visualize_grid PRIVATE /W4 /permissive-)
    target_compile_options(test_game_state PRIVATE /W4 /permissive-)
    target_compile_options(test_move_gen PRIVATE /W4 /permissive-)
    target_compile_options(test_state_transition PRIVATE /W4 /permissive-)
    target_compile_options(test_mcts PRIVATE /W4 /permissive-)
    target_compile_options(test_mcts_agent PRIVATE /W4 /permissive-)
else()
    target_compile_options(catan_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(catan_mcts PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_visualize_grid PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_game_state PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_move_gen PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_state_transition PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_mcts PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_mcts_agent PRIVATE -Wall -Wextra -Wpedantic)
endif()
