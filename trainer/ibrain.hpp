#pragma once

#include "types.hpp"

#include <string>

namespace Aronda::Trainer
{

class IBrain
{
public:
    virtual ~IBrain() = default;

    void save(const std::string& path) const;

    Action predict(const State& current_state) const;

    /**
     * @pre states.size() == actions.size()
     */
    void train(const std::vector<State>& states, const std::vector<Action>& actions);

protected:
    IBrain() = default;

private:
    virtual void impl_save(const std::string& path) const = 0;

    virtual Action impl_predict(const State& current_state) const = 0;

    virtual void impl_train(const std::vector<State>& states, const std::vector<Action>& actions) = 0;
};
}
