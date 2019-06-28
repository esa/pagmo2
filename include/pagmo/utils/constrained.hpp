/* Copyright 2017-2018 PaGMO development team

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

#ifndef PAGMO_CONSTRAINED_HPP
#define PAGMO_CONSTRAINED_HPP

#include <algorithm>
#include <cmath>
#include <utility>

#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

// Tests equality constraints against some tolerance vector. Returns number of constraints satisfied and the L2 norm of
// the violation
template <typename It1, typename It2>
inline std::pair<vector_double::size_type, double> test_eq_constraints(It1 ceq_first, It1 ceq_last, It2 tol_first)
{
    // Main computation
    double l2 = 0.;
    vector_double::size_type n = 0u;
    while (ceq_first != ceq_last) {
        auto err = std::max(std::abs(*ceq_first++) - *tol_first++, 0.);
        l2 += err * err;
        if (err <= 0.) {
            ++n;
        }
    }
    return std::pair<vector_double::size_type, double>(n, std::sqrt(l2));
}

// Tests inequality constraints against some tolerance vector. Returns number of constraints satisfied and the L2 norm
// of the violation
template <typename It1, typename It2>
inline std::pair<vector_double::size_type, double> test_ineq_constraints(It1 cineq_first, It1 cineq_last, It2 tol_first)
{
    // Main computation
    double l2 = 0.;
    vector_double::size_type n = 0u;
    while (cineq_first != cineq_last) {
        auto err = std::max(*cineq_first++ - *tol_first++, 0.);
        l2 += err * err;
        if (err <= 0.) {
            ++n;
        }
    }
    return std::pair<vector_double::size_type, double>(n, std::sqrt(l2));
}

} // namespace detail

// Compares two fitness vectors in a single-objective, constrained, case (from a vector of tolerances)
PAGMO_DLL_PUBLIC bool compare_fc(const vector_double &, const vector_double &, vector_double::size_type,
                                 const vector_double &);

// Compares two fitness vectors in a single-objective, constrained, case (from a scalar tolerance)
PAGMO_DLL_PUBLIC bool compare_fc(const vector_double &, const vector_double &, vector_double::size_type, double);

// Sorts a population in a single-objective, constrained, case (from a vector of tolerances)
PAGMO_DLL_PUBLIC std::vector<pop_size_t> sort_population_con(const std::vector<vector_double> &,
                                                             vector_double::size_type, const vector_double &);

// Sorts a population in a single-objective, constrained, case (from a scalar tolerance)
PAGMO_DLL_PUBLIC std::vector<pop_size_t> sort_population_con(const std::vector<vector_double> &,
                                                             vector_double::size_type, double = 0.);

} // namespace pagmo
#endif
