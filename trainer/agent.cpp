#include "agent.hpp"

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

    auto argmax(const State& state_action)
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

    Action densityToOneHot(Action a)
    {
        //  TODO std::max_element(begin(a), end(a));
        return a;
    }

    namespace Game
    {
        std::tuple<boost::optional<State>, double, bool> play(const State& state, const Action& action)
        {
            return {};
        }

        State begin()
        {
            return {};
        }
    }
}

double Agent::run()
{
    double R = 0.;
    State s = Game::begin();
    while(true)
    {
        const auto a = act(s);
        auto[s_, r, done] = Game::play(s, a);

        if(done) // terminal state
            s_ = boost::none;

        observe({s, a, r, s_});
        replay();

        R += r;
        if(done) return R;
        s = *s_;
    }
}

void Agent::saveModel(const std::string& path) const
{
}

Action Agent::act(const State& state)
{
    if(std::uniform_real_distribution<>(0., 1.)(rng()) < m_epsilon)
        return Utils::oneHotCol<Aronda::State::number_of_square>(
            std::uniform_int_distribution<std::size_t>(0, Aronda::State::number_of_square - 1)(rng()));
    else
        return densityToOneHot(m_brain.predict(state));
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

    const auto p = m_brain.predict(state);
    const auto p_ = m_brain.predict(state_);

    auto[s, a, r, s_] = sample;

    // CNTK : [0] because of sequence dimension
    auto t = p[0];
    if(s_)
        t = r + GAMMA * argmax(p_);
    else
        t = r;

    const auto x = s;
    const auto y = t;

    m_brain.train(x, y);
}
}
