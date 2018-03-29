#pragma once

#include "types.hpp"

#include <boost/optional.hpp>

namespace Aronda::Trainer
{

class IGame
{
public:
    /**
     * if s_ == boost::none, game is done
     */
    struct MoveResult
    {
        boost::optional<State> new_state;
        double reward;
    };

    virtual ~IGame() = default;

    /**
     * @pre action < Aronda::State::number_of_square
     */
    MoveResult play(const State& state, const std::size_t action);

    State begin();

protected:
    IGame() = default;

private:
    virtual MoveResult impl_play(const State& state, const std::size_t action) = 0;

    virtual State impl_begin() = 0;
};
}