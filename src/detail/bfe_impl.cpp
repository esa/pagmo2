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

#include <cassert>
#include <stdexcept>
#include <string>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <pagmo/detail/bfe_impl.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

// Check the input decision vectors for a batch fitness evaluation
// for problem p.
void bfe_check_input_dvs(const problem &p, const vector_double &dvs)
{
    // Fetch the number of dimensions from the problem.
    const auto n_dim = p.get_nx();
    // Get the total number of decision vectors packed in dvs.
    const auto n_dvs = dvs.size() / n_dim;
    // dvs represent a sequence of decision vectors laid out next to each other.
    // Hence, its size must be divided by the problem's dimension exactly.
    if (dvs.size() % n_dim) {
        pagmo_throw(std::invalid_argument, "Invalid argument for a batch fitness evaluation: the length of the vector "
                                           "representing the decision vectors, "
                                               + std::to_string(dvs.size())
                                               + ", is not an exact multiple of the dimension of the problem, "
                                               + std::to_string(n_dim));
    }
    // Check all the decision vectors, using the same function employed
    // in pagmo::problem for dv checking.
    using range_t = tbb::blocked_range<decltype(dvs.size())>;
    tbb::parallel_for(range_t(0, n_dvs), [&p, &dvs, n_dim](const range_t &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
            // NOTE: prob_check_dv only fetches cached data from p,
            // and it is thus thread-safe.
            prob_check_dv(p, dvs.data() + i * n_dim, n_dim);
        }
    });
}

// Check the fitness vectors fvs produced by a bfe for problem p with input
// decision vectors dvs.
void bfe_check_output_fvs(const problem &p, const vector_double &dvs, const vector_double &fvs)
{
    const auto n_dim = p.get_nx();
    const auto n_dvs = dvs.size() / n_dim;
    const auto f_dim = p.get_nf();
    const auto n_fvs = fvs.size() / f_dim;
    // NOTE: assume dvs has been checked already.
    assert(dvs.size() % n_dim == 0u);
    if (fvs.size() % f_dim) {
        // The size of the vector of fitnesses must be divided exactly
        // by the fitness dimension of the problem.
        pagmo_throw(std::invalid_argument,
                    "An invalid result was produced by a batch fitness evaluation: the length of "
                    "the vector representing the fitness vectors, "
                        + std::to_string(fvs.size())
                        + ", is not an exact multiple of the fitness dimension of the problem, "
                        + std::to_string(f_dim));
    }
    if (n_fvs != n_dvs) {
        // The number of fitness vectors produced must be equal to the number of input
        // decision vectors.
        pagmo_throw(
            std::invalid_argument,
            "An invalid result was produced by a batch fitness evaluation: the number of produced fitness vectors, "
                + std::to_string(n_fvs) + ", differs from the number of input decision vectors, "
                + std::to_string(n_dvs));
    }
    // Check all the fitness vectors, using the same function employed
    // in pagmo::problem for fv checking.
    using range_t = tbb::blocked_range<decltype(fvs.size())>;
    tbb::parallel_for(range_t(0, n_fvs), [&p, &fvs, f_dim](const range_t &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
            // NOTE: prob_check_fv only fetches cached data from p,
            // and it is thus thread-safe.
            prob_check_fv(p, fvs.data() + i * f_dim, f_dim);
        }
    });
}

} // namespace detail

} // namespace pagmo
