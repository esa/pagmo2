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

#include <functional>
#include <stdexcept>

#include <pagmo/batch_evaluators/default_bfe.hpp>
#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

namespace
{

// C++ implementation of the heuristic for the automatic deduction of the "best"
// bfe strategy.
vector_double default_bfe_cpp_impl(const problem &p, const vector_double &dvs)
{
    // The member function batch_fitness() of p, if present, has priority.
    if (p.has_batch_fitness()) {
        return member_bfe{}(p, dvs);
    }
    // Otherwise, we run the generic thread-based bfe, if the problem
    // is thread-safe enough.
    if (p.get_thread_safety() >= thread_safety::basic) {
        return thread_bfe{}(p, dvs);
    }
    pagmo_throw(std::invalid_argument,
                "Cannot execute fitness evaluations in batch mode for a problem of type '" + p.get_name()
                    + "': the problem does not implement the batch_fitness() member function, and its thread safety "
                      "level is not sufficient to run a thread-based batch fitness evaluation implementation");
}

} // namespace

std::function<vector_double(const problem &, const vector_double &)> default_bfe_impl = &default_bfe_cpp_impl;

} // namespace detail

// Call operator.
vector_double default_bfe::operator()(const problem &p, const vector_double &dvs) const
{
    return detail::default_bfe_impl(p, dvs);
}

// Serialization support.
template <typename Archive>
void default_bfe::serialize(Archive &, unsigned)
{
}

} // namespace pagmo

PAGMO_S11N_BFE_IMPLEMENT(pagmo::default_bfe)
