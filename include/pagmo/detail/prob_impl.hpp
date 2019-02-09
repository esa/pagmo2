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

#ifndef PAGMO_DETAIL_PROB_IMPL_HPP
#define PAGMO_DETAIL_PROB_IMPL_HPP

#include <stdexcept>
#include <string>

#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>

// NOTE: this header contains various problem-related utilities
// that are used in multiple places.
namespace pagmo
{

namespace detail
{

// Check that the decision vector starting at dv and with
// size s is compatible with the input problem p.
// NOTE: this is templated so that we don't have circular
// include issues - P will always be pagmo::problem.
template <typename P>
inline void prob_check_dv(const P &p, const double *dv, vector_double::size_type s)
{
    (void)dv;
    // 1 - check decision vector for length consistency
    if (s != p.get_nx()) {
        pagmo_throw(std::invalid_argument, "A decision vector is incompatible with a problem of type '" + p.get_name()
                                               + "': the number of dimensions of the problem is "
                                               + std::to_string(p.get_nx())
                                               + ", while the decision vector has a size of " + std::to_string(s)
                                               + " (the two values should be equal)");
    }
    // 2 - Here is where one could check if the decision vector
    // is in the bounds. At the moment not implemented
}

// Check that the fitness vector starting at fv and with size s
// is compatible with the input problem p.
template <typename P>
inline void prob_check_fv(const P &p, const double *fv, vector_double::size_type s)
{
    (void)fv;
    // Checks dimension of returned fitness
    if (s != p.get_nf()) {
        pagmo_throw(std::invalid_argument, "A fitness vector is incompatible with a problem of type '" + p.get_name()
                                               + "': the dimension of the fitness of the problem is "
                                               + std::to_string(p.get_nf())
                                               + ", while the fitness vector has a size of " + std::to_string(s)
                                               + " (the two values should be equal)");
    }
}

} // namespace detail

} // namespace pagmo

#endif
