#pragma once

#include "state/state.hpp"

namespace Aronda::Trainer
{

using State = Aronda::State::Board;

using Action = Utils::Matrix<Aronda::State::number_of_square, 1>;
}