#include "ibrain.hpp"

namespace Aronda::Trainer
{

void IBrain::save(const std::string& path) const
{
    impl_save(path);
}

Action IBrain::predict(const State& current_state) const
{
    return impl_predict(current_state);
}

void IBrain::train(const std::vector<State>& states, const std::vector<Action>& actions)
{
    Expects(states.size() == actions.size());
    impl_train(states, actions);
}
}