#include "agent.hpp"
#include "jaronda.hpp"

#include "cntkbrain.hpp"

#include <iostream>
#include <random>

namespace Aronda::Trainer
{

namespace
{
    auto& rng()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }

    template <class T> auto draw(const T& container)
    {
        Expects(container.size() > 0);
        const auto index = std::uniform_int_distribution<std::size_t>(0, container.size() - 1)(rng());
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
{
    m_brain = std::make_unique<CntkBrain>();
}

double Agent::run()
{
    double R = 0.;
    JAronda game{"11815"};
    std::cout << "new game" << std::endl;
    State s = game.begin();
    while(true)
    {
        std::cout << "new move: \n";
        // std::cout << s << std::endl;
        const auto a = act(s);
        std::cout << "playing " << a << std::endl;
        auto res = game.play(s, a);
        std::cout << "r = " << res.reward << std::endl;
        // if (res.new_state)
        // 	std::cout << *res.new_state << std::endl;

        observe({s, a, res.reward, res.new_state});
        replay();

        R += res.reward;
        if(res.new_state)
            s = *res.new_state;
        else
            return R;
    }
}

void Agent::saveModel(const std::string& path) const
{
    m_brain->save(path);
}

std::size_t Agent::act(const State& state)
{
    if(std::uniform_real_distribution<>(0., 1.)(rng()) < m_epsilon)
        return std::uniform_int_distribution<std::size_t>(0, Aronda::State::number_of_square - 1)(rng());
    else
    {
        const auto qmap = m_brain->predict(state);
        std::cout << "q-map = " << qmap.transpose() << std::endl;
        return argmax(qmap);
    }
}

void Agent::observe(Step step)
{
    m_memory.push_back(std::move(step));

    // slowly decrease Epsilon based on our eperience
    m_steps += 1;
    m_epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * std::exp(-LAMBDA * m_steps);
}

void Agent::replay()
{
    // TODO batch version
    // const auto batch = sample(m_memory, BATCH_SIZE);
    // const auto batchLen = batch.size();
    //
    // const auto no_state = State::Zero();
    //
    // // CNTK: explicitly setting to float32
    // states = numpy.array([o[0] for o in batch], dtype = np.float32);
    // states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch], dtype = np.float32);
    //
    // p = agent.brain.predict(states);
    // p_ = agent.brain.predict(states_);
    //
    // // CNTK: explicitly setting to float32
    // x = numpy.zeros((batchLen, STATE_COUNT)).astype(np.float32);
    // y = numpy.zeros((batchLen, ACTION_COUNT)).astype(np.float32);
    //
    // for(const auto i : range(batchLen))
    // {
    //     auto[s, a, r, s_] batch[i];
    //
    //     // CNTK : [0] because of sequence dimension
    //     auto t = p[0][i];
    //     if(s_ is None)
    //         t[a] = r;
    //     else
    //         t[a] = r + GAMMA * numpy.amax(p_[0][i]);
    //
    //     x[i] = s;
    //     y[i] = t;
    // }
    // m_brain.train(x, y);

    const auto sample = draw(m_memory);

    const auto state = sample.s;
    const auto state_ = sample.s_ ? *sample.s_ : State::Zero();

    const auto p = m_brain->predict(state);
    const auto p_ = m_brain->predict(state_);

    auto[s, a, r, s_] = sample;

    auto t = p;
    if(s_)
        t[a] = r + GAMMA * argmax(p_);
    else
        t[a] = r;

    const auto x = s;
    const auto y = t;

    std::cout << "training : \n";
    std::cout << "\to = " << p.transpose() << std::endl;
    std::cout << "\ty = " << y.transpose() << std::endl;
    m_brain->train(x, y);
}
}
