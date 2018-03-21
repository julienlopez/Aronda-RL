#pragma once

#include "state.hpp"

namespace Aronda::State
{

class Parser
{
public:
    static Board parse(const std::string& json_string);

private:
};
}
