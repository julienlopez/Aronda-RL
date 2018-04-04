#pragma once

#include "types.hpp"

#include <boost/optional.hpp>

namespace Aronda::Trainer
{

class IGame
{
public:
    using GameState = Aronda::State::GameState;

    /**
     * if s_ == boost::none, game is done
     */
    struct MoveResult
    {
        boost::optional<GameState> new_state;
        double reward;
    };

    virtual ~IGame() = default;

    /**
     * @pre action < Aronda::State::number_of_square
     */
    MoveResult play(const State& state, const std::size_t action);

    GameState begin();

protected:
    IGame() = default;

private:
    virtual MoveResult impl_play(const State& state, const std::size_t action) = 0;

    virtual GameState impl_begin() = 0;
};
}