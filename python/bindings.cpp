// bindings.cpp
// Python bindings for Catan RL engine using pybind11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <cstring>

#include "game_state.h"
#include "player_state.h"
#include "action.h"
#include "move_gen.h"
#include "state_transition.h"
#include "state_encoder.h"
#include "mcts/mcts_agent.h"
#include "mcts/alphazero_mcts.h"

namespace py = pybind11;

using namespace catan;
using namespace catan::mcts;
using namespace catan::alphazero;

PYBIND11_MODULE(catan_engine, m) {
    m.doc() = "Catan RL Engine - Fast C++ implementation with Python bindings";

    // ========================================================================
    // Enums
    // ========================================================================
    
    py::enum_<ActionType>(m, "ActionType")
        .value("PlaceInitialSettlement", ActionType::PlaceInitialSettlement)
        .value("PlaceInitialRoad", ActionType::PlaceInitialRoad)
        .value("RollDice", ActionType::RollDice)
        .value("PlaceSettlement", ActionType::PlaceSettlement)
        .value("PlaceRoad", ActionType::PlaceRoad)
        .value("UpgradeToCity", ActionType::UpgradeToCity)
        .value("BuyDevCard", ActionType::BuyDevCard)
        .value("PlayKnight", ActionType::PlayKnight)
        .value("PlayRoadBuilding", ActionType::PlayRoadBuilding)
        .value("PlayYearOfPlenty", ActionType::PlayYearOfPlenty)
        .value("PlayMonopoly", ActionType::PlayMonopoly)
        .value("BankTrade", ActionType::BankTrade)
        .value("PortTrade", ActionType::PortTrade)
        .value("MoveRobber", ActionType::MoveRobber)
        .value("StealFromPlayer", ActionType::StealFromPlayer)
        .value("DiscardResources", ActionType::DiscardResources)
        .value("EndTurn", ActionType::EndTurn)
        .export_values();

    py::enum_<ResourceType>(m, "ResourceType")
        .value("Brick", ResourceType::Brick)
        .value("Lumber", ResourceType::Lumber)
        .value("Wool", ResourceType::Wool)
        .value("Grain", ResourceType::Grain)
        .value("Ore", ResourceType::Ore)
        .export_values();

    py::enum_<GamePhase>(m, "GamePhase")
        .value("Setup", GamePhase::Setup)
        .value("MainGame", GamePhase::MainGame)
        .value("Finished", GamePhase::Finished)
        .export_values();

    py::enum_<TurnPhase>(m, "TurnPhase")
        .value("RollDice", TurnPhase::RollDice)
        .value("Discard", TurnPhase::Discard)
        .value("Robber", TurnPhase::Robber)
        .value("Trading", TurnPhase::Trading)
        .export_values();

    py::enum_<PieceType>(m, "PieceType")
        .value("None_", PieceType::None)
        .value("Road", PieceType::Road)
        .value("Settlement", PieceType::Settlement)
        .value("City", PieceType::City)
        .export_values();

    py::enum_<HarborType>(m, "HarborType")
        .value("None_", HarborType::None)
        .value("Generic", HarborType::Generic)
        .value("Brick", HarborType::Brick)
        .value("Lumber", HarborType::Lumber)
        .value("Wool", HarborType::Wool)
        .value("Grain", HarborType::Grain)
        .value("Ore", HarborType::Ore)
        .export_values();

    py::enum_<Resource>(m, "Resource")
        .value("Desert", Resource::Desert)
        .value("Brick", Resource::Brick)
        .value("Lumber", Resource::Lumber)
        .value("Wool", Resource::Wool)
        .value("Grain", Resource::Grain)
        .value("Ore", Resource::Ore)
        .export_values();

    // ========================================================================
    // Board Structures
    // ========================================================================

    py::class_<Tile>(m, "Tile")
        .def(py::init<>())
        .def_readwrite("resource", &Tile::resource)
        .def_readwrite("number", &Tile::number)
        .def("__repr__", [](const Tile& t) {
            return "Tile(resource=" + std::to_string(static_cast<int>(t.resource)) +
                   ", number=" + std::to_string(static_cast<int>(t.number)) + ")";
        });

    py::class_<Vertex>(m, "Vertex")
        .def(py::init<>())
        .def_readonly("tiles", &Vertex::tiles)
        .def_readonly("edges", &Vertex::edges);

    py::class_<Edge>(m, "Edge")
        .def(py::init<>())
        .def_readonly("vertices", &Edge::vertices)
        .def_readonly("tiles", &Edge::tiles);

    py::class_<Harbor>(m, "Harbor")
        .def(py::init<>())
        .def_readwrite("edge_idx", &Harbor::edge_idx)
        .def_readwrite("type", &Harbor::type);

    py::class_<Piece>(m, "Piece")
        .def(py::init<>())
        .def_readwrite("type", &Piece::type)
        .def_readwrite("owner", &Piece::owner)
        .def("__repr__", [](const Piece& p) {
            return "Piece(type=" + std::to_string(static_cast<int>(p.type)) +
                   ", owner=" + std::to_string(static_cast<int>(p.owner)) + ")";
        });

    py::class_<BoardGrid>(m, "BoardGrid")
        .def(py::init<>())
        .def_readonly("tiles", &BoardGrid::tiles)
        .def_readonly("vertices", &BoardGrid::vertices)
        .def_readonly("edges", &BoardGrid::edges)
        .def_readonly("harbors", &BoardGrid::harbors)
        .def_static("create_default", &BoardGrid::create_default)
        .def("randomize", &BoardGrid::randomize);

    py::class_<PlayerState>(m, "PlayerState")
        .def(py::init<>())
        .def_readonly("resources", &PlayerState::resources)
        .def_readonly("dev_cards", &PlayerState::dev_cards)
        .def_readwrite("settlements_remaining", &PlayerState::settlements_remaining)
        .def_readwrite("cities_remaining", &PlayerState::cities_remaining)
        .def_readwrite("roads_remaining", &PlayerState::roads_remaining)
        .def_readwrite("public_victory_points", &PlayerState::public_victory_points)
        .def_readwrite("hidden_victory_points", &PlayerState::hidden_victory_points)
        .def_readwrite("knights_played", &PlayerState::knights_played)
        .def_readwrite("has_longest_road", &PlayerState::has_longest_road)
        .def_readwrite("has_largest_army", &PlayerState::has_largest_army)
        .def("total_victory_points", &PlayerState::total_victory_points)
        .def("total_resources", &PlayerState::total_resources)
        .def("__repr__", [](const PlayerState& p) {
            return "PlayerState(vp=" + std::to_string(p.total_victory_points()) +
                   ", resources=" + std::to_string(p.total_resources()) + ")";
        });

    // ========================================================================
    // Action
    // ========================================================================
    
    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("type", &Action::type)
        .def_readwrite("location", &Action::location)
        .def_readwrite("param1", &Action::param1)
        .def_readwrite("param2", &Action::param2)
        .def_readwrite("param3", &Action::param3)
        .def_readwrite("param4", &Action::param4)
        .def_static("place_initial_settlement", &Action::place_initial_settlement)
        .def_static("place_initial_road", &Action::place_initial_road)
        .def_static("roll_dice", &Action::roll_dice)
        .def_static("place_settlement", &Action::place_settlement)
        .def_static("place_road", &Action::place_road)
        .def_static("upgrade_to_city", &Action::upgrade_to_city)
        .def_static("buy_dev_card", &Action::buy_dev_card)
        .def_static("bank_trade", &Action::bank_trade)
        .def_static("port_trade", &Action::port_trade)
        .def_static("end_turn", &Action::end_turn)
        .def("__repr__", [](const Action& a) {
            return "Action(type=" + std::to_string(static_cast<int>(a.type)) + 
                   ", location=" + std::to_string(static_cast<int>(a.location)) + ")";
        });

    // ========================================================================
    // GameState
    // ========================================================================
    
    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def_static("create_new_game", &GameState::create_new_game,
            py::arg("num_players") = 4,
            py::arg("seed") = 0,
            "Create a new game with specified number of players and random seed")
        .def_readwrite("num_players", &GameState::num_players)
        .def_readwrite("current_player", &GameState::current_player)
        .def_readwrite("game_phase", &GameState::game_phase)
        .def_readwrite("turn_phase", &GameState::turn_phase)
        .def_readwrite("turn_number", &GameState::turn_number)
        .def_readwrite("last_dice_roll", &GameState::last_dice_roll)
        .def_readwrite("robber_tile", &GameState::robber_tile)
        .def_readwrite("longest_road_owner", &GameState::longest_road_owner)
        .def_readwrite("longest_road_length", &GameState::longest_road_length)
        .def_readwrite("largest_army_owner", &GameState::largest_army_owner)
        .def_readwrite("largest_army_count", &GameState::largest_army_count)
        .def_readonly("board", &GameState::board,
            "Board grid (tiles, vertices, edges, harbors)")
        .def_readonly("players", &GameState::players,
            "Array of player states")
        .def_readonly("vertex_pieces", &GameState::vertex_pieces,
            "Pieces placed on vertices (settlements/cities)")
        .def_readonly("edge_pieces", &GameState::edge_pieces,
            "Pieces placed on edges (roads)")
        .def_readonly("dev_deck", &GameState::dev_deck,
            "Development cards remaining in deck")
        .def_readonly("resource_bank", &GameState::resource_bank,
            "Resources remaining in bank")
        .def("is_game_over", &GameState::is_game_over)
        .def("get_winner", &GameState::get_winner)
        .def("is_vertex_occupied", &GameState::is_vertex_occupied)
        .def("is_edge_occupied", &GameState::is_edge_occupied)
        .def("has_harbor_access", &GameState::has_harbor_access)
        .def("get_trade_ratio", &GameState::get_trade_ratio)
        .def("get_player_vp", [](const GameState& state, std::uint8_t player_idx) {
            if (player_idx >= state.num_players) return 0;
            return static_cast<int>(state.players[player_idx].total_victory_points());
        }, "Get victory points for a player")
        .def("get_player_resources", [](const GameState& state, std::uint8_t player_idx) {
            if (player_idx >= state.num_players) return py::array_t<int>();
            const auto& resources = state.players[player_idx].resources;
            return py::array_t<int>(
                {NUM_RESOURCE_TYPES},
                {sizeof(std::uint8_t)},
                reinterpret_cast<const int*>(resources.data())
            );
        }, "Get resource counts for a player")
        .def("copy", [](const GameState& state) {
            return GameState(state);
        }, "Create a copy of the game state")
        .def("__repr__", [](const GameState& state) {
            return "GameState(players=" + std::to_string(state.num_players) +
                   ", turn=" + std::to_string(state.turn_number) +
                   ", phase=" + (state.game_phase == GamePhase::Setup ? "Setup" :
                                 state.game_phase == GamePhase::MainGame ? "MainGame" : "Finished") + ")";
        });

    // ========================================================================
    // Move Generation and State Transition
    // ========================================================================
    
    m.def("generate_legal_actions", 
        [](const GameState& state) {
            std::vector<Action> actions;
            generate_legal_actions(state, actions);
            return actions;
        },
        py::arg("state"),
        "Generate all legal actions from current state");

    m.def("apply_action",
        [](GameState& state, const Action& action, std::uint32_t rng_seed) {
            return apply_action(state, action, rng_seed);
        },
        py::arg("state"),
        py::arg("action"),
        py::arg("rng_seed") = 0,
        "Apply action to game state (mutates state in-place)");

    // ========================================================================
    // MCTS Configuration
    // ========================================================================
    
    py::class_<MCTSConfig>(m, "MCTSConfig")
        .def(py::init<>())
        .def_readwrite("num_iterations", &MCTSConfig::num_iterations)
        .def_readwrite("exploration_constant", &MCTSConfig::exploration_constant)
        .def_readwrite("max_rollout_depth", &MCTSConfig::max_rollout_depth)
        .def_readwrite("random_seed", &MCTSConfig::random_seed)
        .def_readwrite("use_visit_count", &MCTSConfig::use_visit_count)
        .def("__repr__", [](const MCTSConfig& cfg) {
            return "MCTSConfig(iterations=" + std::to_string(cfg.num_iterations) +
                   ", c=" + std::to_string(cfg.exploration_constant) + ")";
        });

    // ========================================================================
    // Action Policy (for training data)
    // ========================================================================
    
    py::class_<ActionPolicy>(m, "ActionPolicy")
        .def(py::init<>())
        .def_readwrite("action", &ActionPolicy::action)
        .def_readwrite("probability", &ActionPolicy::probability)
        .def_readwrite("visits", &ActionPolicy::visits)
        .def("__repr__", [](const ActionPolicy& ap) {
            return "ActionPolicy(prob=" + std::to_string(ap.probability) +
                   ", visits=" + std::to_string(ap.visits) + ")";
        });

    // ========================================================================
    // MCTS Agent
    // ========================================================================
    
    py::class_<MCTSAgent>(m, "MCTSAgent")
        .def(py::init<const MCTSConfig&>(),
            py::arg("config") = MCTSConfig(),
            "Create MCTS agent with configuration")
        .def("select_action", &MCTSAgent::select_action,
            py::arg("state"),
            "Select best action from game state")
        .def("get_action_policy", &MCTSAgent::get_action_policy,
            py::arg("state"),
            "Get action policy (probabilities) from MCTS search")
        .def("play_game", &MCTSAgent::play_game,
            py::arg("state"),
            py::arg("max_actions") = 500,
            "Play full game and return winner (0xFF if incomplete)")
        .def("get_config", &MCTSAgent::get_config,
            py::return_value_policy::reference_internal)
        .def("set_config", &MCTSAgent::set_config,
            py::arg("config"));

    // ========================================================================
    // Self-Play Statistics
    // ========================================================================
    
    py::class_<SelfPlayStats>(m, "SelfPlayStats")
        .def(py::init<>())
        .def_readwrite("games_played", &SelfPlayStats::games_played)
        .def_readwrite("games_finished", &SelfPlayStats::games_finished)
        .def_readwrite("wins_by_player", &SelfPlayStats::wins_by_player)
        .def_readwrite("avg_game_length", &SelfPlayStats::avg_game_length)
        .def_readwrite("avg_final_vp", &SelfPlayStats::avg_final_vp)
        .def("__repr__", [](const SelfPlayStats& stats) {
            return "SelfPlayStats(played=" + std::to_string(stats.games_played) +
                   ", finished=" + std::to_string(stats.games_finished) + ")";
        });

    // ========================================================================
    // Self-Play Engine
    // ========================================================================
    
    py::class_<SelfPlayEngine>(m, "SelfPlayEngine")
        .def(py::init<const MCTSConfig&>(),
            py::arg("config") = MCTSConfig(),
            "Create self-play engine with MCTS configuration")
        .def("play_games", &SelfPlayEngine::play_games,
            py::arg("num_games"),
            py::arg("num_players") = 4,
            py::arg("seed") = 0,
            "Play multiple games and return statistics")
        .def("get_config", &SelfPlayEngine::get_config,
            py::return_value_policy::reference_internal)
        .def("set_config", &SelfPlayEngine::set_config,
            py::arg("config"));

    // ========================================================================
    // AlphaZero MCTS
    // ========================================================================
    
    py::class_<NNEvaluation>(m, "NNEvaluation")
        .def(py::init<>())
        .def_readwrite("policy", &NNEvaluation::policy,
            "Policy vector (prior probabilities for each legal action)")
        .def_readwrite("value", &NNEvaluation::value,
            "State value from current player's perspective [-1, 1]")
        .def("__repr__", [](const NNEvaluation& eval) {
            return "NNEvaluation(policy_size=" + std::to_string(eval.policy.size()) +
                   ", value=" + std::to_string(eval.value) + ")";
        });
    
    py::class_<AlphaZeroConfig>(m, "AlphaZeroConfig")
        .def(py::init<>())
        .def_readwrite("num_simulations", &AlphaZeroConfig::num_simulations,
            "Number of MCTS simulations per move")
        .def_readwrite("cpuct", &AlphaZeroConfig::cpuct,
            "PUCT exploration constant")
        .def_readwrite("dirichlet_alpha", &AlphaZeroConfig::dirichlet_alpha,
            "Dirichlet noise alpha for root exploration")
        .def_readwrite("dirichlet_weight", &AlphaZeroConfig::dirichlet_weight,
            "Weight of Dirichlet noise at root")
        .def_readwrite("add_exploration_noise", &AlphaZeroConfig::add_exploration_noise,
            "Whether to add exploration noise at root")
        .def_readwrite("random_seed", &AlphaZeroConfig::random_seed,
            "Random seed (0 = time-based)")
        .def("__repr__", [](const AlphaZeroConfig& cfg) {
            return "AlphaZeroConfig(sims=" + std::to_string(cfg.num_simulations) +
                   ", cpuct=" + std::to_string(cfg.cpuct) + ")";
        });
    
    py::class_<AlphaZeroMCTS>(m, "AlphaZeroMCTS")
        .def(py::init<const AlphaZeroConfig&, NNEvaluator>(),
            py::arg("config"),
            py::arg("evaluator"),
            "Create AlphaZero MCTS with configuration and NN evaluator")
        .def("search", &AlphaZeroMCTS::search,
            py::arg("state"),
            "Run MCTS search and return best action")
        .def("get_action_probabilities", &AlphaZeroMCTS::get_action_probabilities,
            "Get action visit count probabilities (for training data)");

    // ========================================================================
    // Free Functions
    // ========================================================================
    
    m.def("generate_legal_actions", 
        py::overload_cast<const GameState&, std::vector<Action>&>(&generate_legal_actions),
        py::arg("state"),
        py::arg("out_actions"),
        "Generate legal actions for game state (mutates out_actions vector)");
    
    m.def("generate_legal_actions",
        [](const GameState& state) {
            std::vector<Action> actions;
            generate_legal_actions(state, actions);
            return actions;
        },
        py::arg("state"),
        "Generate legal actions for game state (returns new vector)");
    
    m.def("apply_action",
        &apply_action,
        py::arg("state"),
        py::arg("action"),
        py::arg("rng_seed"),
        "Apply action to game state (mutates state in-place)");
    
    // ========================================================================
    // State Encoder
    // ========================================================================
    
    py::class_<StateEncoder>(m, "StateEncoder")
        .def(py::init<>(),
            "Create state encoder")
        .def("encode_state",
            [](const StateEncoder& encoder, const GameState& state, std::uint8_t perspective_player) {
                // Encode to std::vector<float>
                std::vector<float> features = encoder.encode_state(state, perspective_player);
                
                // Convert to numpy array (zero-copy view)
                return py::array_t<float>(
                    {static_cast<py::ssize_t>(features.size())},  // shape
                    {sizeof(float)},  // strides
                    features.data(),  // data pointer
                    py::cast(features)  // keep vector alive
                );
            },
            py::arg("state"),
            py::arg("perspective_player"),
            "Encode game state to feature vector from perspective player's view")
        .def_static("get_feature_size", &StateEncoder::get_feature_size,
            "Get total number of features in encoded state");
}

