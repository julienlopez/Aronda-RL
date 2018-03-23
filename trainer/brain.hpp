#pragma once

#include "types.hpp"

#include <string>

namespace Aronda::Trainer
{

class Brain
{
public:
    Brain() = default;
    ~Brain() = default;

    void save(const std::string& path) const;

    Action predict(const State& current_state) const;

    void train(const State& state, const Action& action);
};
}
