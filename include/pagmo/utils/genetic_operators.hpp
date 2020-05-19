/* Copyright 2017-2020 PaGMO development team

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

#ifndef PAGMO_UTILS_GENETIC_OPERATORS_HPP
#define PAGMO_UTILS_GENETIC_OPERATORS_HPP

#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{
std::pair<vector_double, vector_double> sbx_crossover_impl(const vector_double &, const vector_double &,
                                                           const std::pair<vector_double, vector_double> &,
                                                           vector_double::size_type, const double, const double,
                                                           detail::random_engine_type &);

void polynomial_mutation_impl(vector_double &, const std::pair<vector_double, vector_double> &,
                              vector_double::size_type, const double, const double, detail::random_engine_type &);

vector_double::size_type mo_tournament_selection_impl(vector_double::size_type, vector_double::size_type,
                                                      const std::vector<vector_double::size_type> &,
                                                      const std::vector<double> &, detail::random_engine_type &);

} // namespace detail

PAGMO_DLL_PUBLIC std::pair<vector_double, vector_double> sbx_crossover(const vector_double &, const vector_double &,
                                                                       const std::pair<vector_double, vector_double> &,
                                                                       vector_double::size_type, const double,
                                                                       const double, detail::random_engine_type &);

PAGMO_DLL_PUBLIC void polynomial_mutation(vector_double &, const std::pair<vector_double, vector_double> &,
                                          vector_double::size_type, const double, const double,
                                          detail::random_engine_type &);

} // namespace pagmo

#endif
