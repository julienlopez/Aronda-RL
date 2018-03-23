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
template <int nb_rows, int nb_cols> Matrix<nb_rows, nb_cols> oneHot(const std::size_t index)
{
    static_assert(nb_rows == 1 || nb_cols == 1);
    const auto state_size = (nb_rows == 1 ? nb_cols : nb_rows);
    Expects(index < state_size);
    Eigen::Array<float, 1, state_size> res;
    for(std::size_t i = 0; i < state_size; i++)
        res(i) = ((i == index) ? 1.f : 0.f);
    return res.matrix();
}

template <int nb_cols> auto oneHotRow(const std::size_t index)
{
    return oneHot<1, nb_cols>(index);
}

template <int nb_rows> auto oneHotCol(const std::size_t index)
{
    return oneHot<nb_rows, 1>(index);
}
}