/* Copyright 2017 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#ifndef PAGMO_UTILS_GENERIC_HPP
#define PAGMO_UTILS_GENERIC_HPP

/** \file gradients_and_hessians.hpp
 * \brief Utilities of general interest for gradients and hessians related calculations
 *
 * This header contains utilities useful in general for gradients and hessians related calculations
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "../exceptions.hpp"
#include "../problem.hpp"
#include "../types.hpp"

namespace pagmo
{
/// Heuristics to estimate the sparsity pattern of some fitness function
/**
 * A numerical estimation of the sparsity pattern of a given pagmo::problem is made by numerically
 * computing the fitness around a given decision vector and detecting the components that are changed.
 *
 * The procedure is clearly risky as its a necessary condition not sufficient to guarantee the correctness of the
 * result. It is of use, though, in tests or cases where its extremely difficult to write the sparsity down and one
 * still wants an estimate.
 *
 * @tparam Func a callable object assumed to be returning a fitness vector when called on \p x
 * @param f instance of the callable object
 * @param x decision vector to test the sparisty around
 * @param dx percentual variation on the decision vector component to detect the sparsity from
 * @return the sparsity_pattern of \p f as detected around \p x
 *
 * @throw std::invalid_argument if \p f returns fitness vecors of different sizes when perturbing \p x
 */
template <typename Func>
sparsity_pattern estimate_sparsity(Func fitness, const vector_double &x, double dx = 1e-8)
{
    vector_double f0 = fitness(x);
    vector_double x_new = x;
    sparsity_pattern retval;
    // We change one by one each variable by dx and detect changes in the fitness
    for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
        x_new[j] = x[j] + std::max(std::abs(x[j]), 1.0) * 1e-8;
        auto f_new = fitness(x_new);
        if (f_new.size() != f0.size()) {
            pagmo_throw(std::invalid_argument,
                        "Change in fitness size detected around the reference point. Cannot estimate a sparisty.");
        }
        for (decltype(f_new.size()) i = 0u; i < f_new.size(); ++i) {
            if (f_new[i] != f0[i]) {
                retval.push_back({i, j});
            }
        }
        x_new[j] = x[j];
    }
    // Restore the lexicographic order required by pagmo::problem::gradient_sparsity
    std::sort(retval.begin(), retval.end());
    return retval;
}

/**
template <typename Func>
sparsity_pattern estimate_gradient3(Func fitness, const vector_double &x, double dx = 1e-8)
{
    auto f0 = fitness(x);
    vector_double gradient(f0.size() * x.size(), 0.);
    // We change one by one each variable by dx and estimate the derivative
    for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
        const double dx1 = std::max(std::abs(x[j]), 1.0) * dx;
        const double dx2 = dx1 * 2;
        const double dx3 = dx1 * 3;
        x_new1[j] = x[j] + dx1;
        x_new2[j] = x[j] + dx2;
        x_new3[j] = x[j] + dx3;
        x_new1[j] = x[j] - dx1;
        x_new2[j] = x[j] - dx2;
        x_new3[j] = x[j] - dx3;

        auto f_new = fitness(x_new);
        if (f_new.size() != f0.size()) {
            pagmo_throw(std::invalid_argument,
                        "Change in fitness size detected around the reference point. Cannot compute a gradient");
        }
        for (decltype(f_new.size()) i = 0u; i < f_new.size(); ++i) {
            const double m1 = (my_func(x + dx1) - my_func(x - dx1)) / 2;
            const double m2 = (my_func(x + dx2) - my_func(x - dx2)) / 4;
            const double m3 = (my_func(x + dx3) - my_func(x - dx3)) / 6;
        }
        x_new[j] = x[j];
    }
}

template <typename Func>
double estimate_derivative3(Func my_func, const double x, const double dx = 1e-8)
{
    // Compute d/dx[func(*first)] using a three-point
    // central difference rule of O(dx^6).
    // From
    // http://www.boost.org/doc/libs/1_55_0/libs/multiprecision/doc/html/boost_multiprecision/tut/floats/fp_eg/nd.html

    const double dx1 = dx;
    const double dx2 = dx1 * 2;
    const double dx3 = dx1 * 3;

    const double m1 = (my_func(x + dx1) - my_func(x - dx1)) / 2;
    const double m2 = (my_func(x + dx2) - my_func(x - dx2)) / 4;
    const double m3 = (my_func(x + dx3) - my_func(x - dx3)) / 6;

    const double fifteen_m1 = 15 * m1;
    const double six_m2 = 6 * m2;
    const double ten_dx1 = 10 * dx1;

    return ((fifteen_m1 - six_m2) + m3) / ten_dx1;
}*/
}
// namespace pagmo

#endif
