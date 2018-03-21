#pragma once

#include <gsl/gsl_assert>

#include <Eigen/Dense>

namespace Aronda::Utils
{
template <int ROWS, int COLS> using Matrix = Eigen::Matrix<float, ROWS, COLS>;

/**
 * @brief returns a 1-hot vector (zero everywhere except for on index set to one) of size 'state_size'
 * @pre index < state_size
 */
template <std::size_t state_size> Matrix<1, state_size> oneHot(const std::size_t index)
{
    Expects(index < state_size);
    Eigen::Array<float, 1, state_size> res;
    for(std::size_t i = 0; i < state_size; i++)
        res(i) = ((i == index) ? 1.f : 0.f);
    return res.matrix();
}
}