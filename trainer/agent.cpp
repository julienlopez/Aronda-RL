#include "agent.hpp"
#include "cntkbrain.hpp"
#include "random.hpp"

namespace Aronda::Trainer
{

namespace
{
    template <class T> auto draw(const T& container)
    {
        Expects(container.size() > 0);
        const auto index = Random::uniform<std::size_t>(0, container.size() - 1);
        return container[index];
    }

    auto argmax(const Action& state_action)
    {
        std::vector<int> max_index_list;
        auto max_value = state_action[0];
        for(int index = 0; index < state_action.rows(); index++)
        {
            const auto value = state_action[index];
            if(value > max_value)
            {
                max_index_list.clear();
                max_value = value;
                max_index_list.push_back(index);
            }
            else if(value == max_value)
                max_index_list.push_back(index);
        }
        return draw(max_index_list);
    }
}

Agent::Agent()
    : m_brain(std::make_unique<CntkBrain>())
    , m_memory(MEMORY_CAPACITY)
{
}

void Agent::saveModel(const std::string& path) const
{
    m_brain->save(path);
}

std::size_t Agent::act(const State& state) const
{
    if (std::uniform_real_distribution<>(0., 1.)(Random::rng()) < m_epsilon)
        return Random::uniform<std::size_t>(0, Aronda::State::number_of_square - 1);
    else
    {
        const auto qmap = m_brain->predict(state);
        return argmax(qmap);
    }
}

void Agent::observe(Step step)
{
    m_memory.add(std::move(step));
    // slowly decrease Epsilon based on our eperience
    m_steps += 1;
    m_epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * std::exp(-LAMBDA * m_steps);
}

void Agent::replay()
{
    const auto batch = m_memory.sample(BATCH_SIZE);
    std::vector<State> x(batch.size());
    std::vector<Action> y(batch.size());
    for (std::size_t i = 0; i < batch.size(); i++)
    {
        const auto sample = batch[i];
    
        const auto state = sample.s;
        const auto state_ = sample.s_ ? *sample.s_ : State::Zero();
    
        const auto p = m_brain->predict(state);
        const auto p_ = m_brain->predict(state_);
    
        auto[s, a, r, s_] = sample;
    
        auto t = p;
        if (s_)
            t[a] = r + GAMMA * argmax(p_);
        else
            t[a] = r;
    
        x[i] = s;
        y[i] = t;
    }
    m_brain->train(x, y);
}

double Agent::epsilon() const
{
    return m_epsilon;
}
}
