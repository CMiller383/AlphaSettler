// action.h
// Compact action representation for Catan RL engine.
// Used for move generation, MCTS tree edges, and state transitions.

#pragma once

#include <cstdint>
#include "resources.h"
#include "dev_cards.h"

namespace catan {

// All possible action types in Catan.
// Designed for fast switch/case dispatch in hot loops.
enum class ActionType : std::uint8_t {
    // Setup phase actions
    PlaceInitialSettlement = 0,
    PlaceInitialRoad,

    // Main game actions
    RollDice,
    PlaceSettlement,
    PlaceRoad,
    UpgradeToCity,
    BuyDevCard,
    
    // Development card plays
    PlayKnight,
    PlayRoadBuilding,
    PlayYearOfPlenty,
    PlayMonopoly,

    // Trading actions
    BankTrade,         // 4:1 trade with bank
    PortTrade,         // 3:1 or 2:1 trade with port

    // Robber actions
    MoveRobber,
    StealFromPlayer,

    // Discard (when rolling 7 with >7 cards)
    DiscardResources,

    // Turn management
    EndTurn,

    // Sentinel
    COUNT
};

// Compact action representation.
// All actions fit in 8 bytes for cache efficiency.
struct Action {
    ActionType type{ActionType::COUNT};

    // Action-specific parameters (interpretation depends on type)
    // For placement: location is vertex/edge/tile index
    // For trades: param1/param2 encode resources
    // For dev cards: param1 may encode additional info
    std::uint8_t location{0};
    std::uint8_t param1{0};
    std::uint8_t param2{0};
    std::uint8_t param3{0};
    std::uint8_t param4{0};
    
    std::uint8_t _padding[2]{0, 0}; // Align to 8 bytes

    // Factory methods for common actions
    static Action place_settlement(std::uint8_t vertex_idx) {
        Action a;
        a.type = ActionType::PlaceSettlement;
        a.location = vertex_idx;
        return a;
    }

    static Action place_initial_settlement(std::uint8_t vertex_idx) {
        Action a;
        a.type = ActionType::PlaceInitialSettlement;
        a.location = vertex_idx;
        return a;
    }

    static Action place_road(std::uint8_t edge_idx) {
        Action a;
        a.type = ActionType::PlaceRoad;
        a.location = edge_idx;
        return a;
    }

    static Action place_initial_road(std::uint8_t edge_idx) {
        Action a;
        a.type = ActionType::PlaceInitialRoad;
        a.location = edge_idx;
        return a;
    }

    static Action upgrade_to_city(std::uint8_t vertex_idx) {
        Action a;
        a.type = ActionType::UpgradeToCity;
        a.location = vertex_idx;
        return a;
    }

    static Action buy_dev_card() {
        Action a;
        a.type = ActionType::BuyDevCard;
        return a;
    }

    static Action play_knight(std::uint8_t target_tile, std::uint8_t steal_from_player) {
        Action a;
        a.type = ActionType::PlayKnight;
        a.location = target_tile;
        a.param1 = steal_from_player;
        return a;
    }

    static Action move_robber(std::uint8_t target_tile) {
        Action a;
        a.type = ActionType::MoveRobber;
        a.location = target_tile;
        return a;
    }

    static Action steal_from_player(std::uint8_t player_idx) {
        Action a;
        a.type = ActionType::StealFromPlayer;
        a.param1 = player_idx;
        return a;
    }

    // Bank trade: give 4 of one resource, get 1 of another
    static Action bank_trade(ResourceType give, ResourceType receive) {
        Action a;
        a.type = ActionType::BankTrade;
        a.param1 = static_cast<std::uint8_t>(give);
        a.param2 = static_cast<std::uint8_t>(receive);
        return a;
    }

    // Port trade: give N of one resource (2 or 3), get 1 of another
    static Action port_trade(ResourceType give, ResourceType receive, std::uint8_t give_count) {
        Action a;
        a.type = ActionType::PortTrade;
        a.param1 = static_cast<std::uint8_t>(give);
        a.param2 = static_cast<std::uint8_t>(receive);
        a.param3 = give_count;
        return a;
    }

    static Action play_year_of_plenty(ResourceType res1, ResourceType res2) {
        Action a;
        a.type = ActionType::PlayYearOfPlenty;
        a.param1 = static_cast<std::uint8_t>(res1);
        a.param2 = static_cast<std::uint8_t>(res2);
        return a;
    }

    static Action play_monopoly(ResourceType resource) {
        Action a;
        a.type = ActionType::PlayMonopoly;
        a.param1 = static_cast<std::uint8_t>(resource);
        return a;
    }

    static Action play_road_building(std::uint8_t edge1, std::uint8_t edge2) {
        Action a;
        a.type = ActionType::PlayRoadBuilding;
        a.param1 = edge1;
        a.param2 = edge2;
        return a;
    }

    // Discard: encode resources to discard in param1-param4 (counts per resource type)
    static Action discard_resources(std::uint8_t brick, std::uint8_t lumber, 
                                    std::uint8_t wool, std::uint8_t grain, std::uint8_t ore) {
        Action a;
        a.type = ActionType::DiscardResources;
        // Pack into params (first 5 resource types)
        a.param1 = brick;
        a.param2 = lumber;
        a.param3 = wool;
        a.param4 = grain;
        a.location = ore; // Reuse location field for 5th resource
        return a;
    }

    static Action roll_dice() {
        Action a;
        a.type = ActionType::RollDice;
        return a;
    }

    static Action end_turn() {
        Action a;
        a.type = ActionType::EndTurn;
        return a;
    }

    // Equality for testing/debugging
    bool operator==(const Action& other) const {
        return type == other.type &&
               location == other.location &&
               param1 == other.param1 &&
               param2 == other.param2 &&
               param3 == other.param3 &&
               param4 == other.param4;
    }

    bool operator!=(const Action& other) const {
        return !(*this == other);
    }
};

static_assert(sizeof(Action) == 8, "Action must be 8 bytes for cache efficiency");

} // namespace catan
