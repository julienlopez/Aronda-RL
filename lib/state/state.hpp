#pragma once

#include "utils/matrix.hpp"

#include <boost/optional.hpp>

namespace Aronda::State
{
static const std::size_t number_of_square = 25;

static const std::size_t number_of_state_per_square = 27;

using Square = Utils::Matrix<1, number_of_state_per_square>;

using Board = Utils::Matrix<number_of_square, number_of_state_per_square>;

enum class Player
{
    Black,
    White
};

struct GameState
{
    Player current_player;
    Board board;
    boost::optional<Player> winner;
};
}
