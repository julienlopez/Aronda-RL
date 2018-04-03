#pragma once

#include "ibrain.hpp"

#include <memory>
#include <vector>

#include <boost/optional.hpp>

namespace Aronda::Trainer
{

class Agent
{
public:
    struct Step
    {
        State s;
        std::size_t a; // index of an Action to form 1-hot action vector
        double r;
        boost::optional<State> s_;
    };

    static constexpr double GAMMA = 0.99; // discount factor

    static constexpr double MAX_EPSILON = 0.9; // 1
    static constexpr double MIN_EPSILON = 0.01; // stay a bit curious even when getting old
    static constexpr double LAMBDA = 0.0001; // speed of decay

    static constexpr std::size_t BATCH_SIZE = 64;

    Agent();

    ~Agent() = default;

    double run();

    void saveModel(const std::string& path) const;

    Action test() const;

private:
    std::unique_ptr<IBrain> m_brain;
    std::vector<Step> m_memory;
    std::size_t m_steps = 0;
    double m_epsilon = MAX_EPSILON;

    std::size_t act(const State& s) const;

    void observe(Step step);

    void replay();
};
}
