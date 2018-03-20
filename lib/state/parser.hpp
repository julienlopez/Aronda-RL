#pragma once

#include "state.hpp"

namespace Aronda::State
{

class Parser
{
public:
    static State parse(const std::string& json_string);

private:
};
}
