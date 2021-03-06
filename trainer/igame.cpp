#include "igame.hpp"

#include <gsl/gsl_assert>

namespace Aronda::Trainer
{

auto IGame::play(const State& state, const std::size_t action) -> MoveResult
{
    Expects(action < Aronda::State::number_of_square);
    return impl_play(state, action);
}

auto IGame::begin() -> GameState
{
    return impl_begin();
}
}
