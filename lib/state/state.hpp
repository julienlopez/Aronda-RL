#pragma once

#include <Eigen/Dense>

namespace Aronda::State
{
static const std::size_t number_of_square = 25;

static const std::size_t number_of_state_per_square = 27;

using Square = Eigen::Matrix<float, 1, number_of_state_per_square>;

using Board = Eigen::Matrix<float, number_of_square, number_of_state_per_square>;
}
