#pragma once

#include "state.hpp"

#include <nlohmann_json/json.hpp>

namespace Aronda::State
{

class Parser
{
public:
    static GameState parse(const std::string& json_string);

    static Square parseSquare(const nlohmann::json& square, const Player current_player);
};
}
