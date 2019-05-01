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

#include <algorithm>
#include <cassert>
#include <iterator>
#include <limits>
#include <stdexcept>

#include <boost/numeric/conversion/cast.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

// Call operator.
vector_double thread_bfe::operator()(const problem &p, const vector_double &dvs) const
{
    // Fetch a few quantities from the problem.
    // Problem dimension.
    const auto n_dim = p.get_nx();
    // Fitness dimension.
    const auto f_dim = p.get_nf();
    // Total number of dvs.
    const auto n_dvs = dvs.size() / n_dim;

    // NOTE: as usual, we assume that thread_bfe is always wrapped
    // by a bfe, where we already check that dvs
    // is compatible with p.
    // NOTE: this is what we always do with user-defined classes:
    // we do the sanity checks in the type-erased container.
    assert(dvs.size() % n_dim == 0u);

    // Prepare the return value.
    // Guard against overflow.
    // LCOV_EXCL_START
    if (n_dvs > std::numeric_limits<vector_double::size_type>::max() / f_dim) {
        pagmo_throw(std::overflow_error,
                    "Overflow detected in the computation of the size of the output of a thread_bfe");
    }
    // LCOV_EXCL_STOP
    vector_double retval(n_dvs * f_dim);

    // Functor to implement the fitness evaluation of a range of input dvs. begin/end are the indices
    // of the individuals in dv (ranging from 0 to n_dvs), the resulting fitnesses will be written directly into
    // retval.
    auto range_evaluator = [&dvs, &retval, n_dim, f_dim, n_dvs](const problem &prob, decltype(dvs.size()) begin,
                                                                decltype(dvs.size()) end) {
        assert(begin <= end);
        assert(end <= n_dvs);
        (void)n_dvs;

        // Temporary dv that will be used for fitness evaluation.
        vector_double tmp_dv(n_dim);
        for (; begin != end; ++begin) {
            auto in_ptr = dvs.data() + begin * n_dim;
            auto out_ptr = retval.data() + begin * f_dim;
            std::copy(
#if defined(_MSC_VER)
                stdext::make_checked_array_iterator(in_ptr, n_dim),
                stdext::make_checked_array_iterator(in_ptr, n_dim, n_dim), tmp_dv.begin()
#else
                in_ptr, in_ptr + n_dim, tmp_dv.begin()
#endif
            );
            const auto fv = prob.fitness(tmp_dv);
            assert(fv.size() == f_dim);
            std::copy(
#if defined(_MSC_VER)
                fv.begin(), fv.end(), stdext::make_checked_array_iterator(out_ptr, f_dim)
#else
                fv.begin(), fv.end(), out_ptr
#endif
            );
        }
    };

    using range_t = tbb::blocked_range<decltype(dvs.size())>;
    if (p.get_thread_safety() >= thread_safety::constant) {
        // We can concurrently call the objfun on the input prob, hence we can
        // capture it by reference and do all the fitness calls on the same object.
        tbb::parallel_for(range_t(0u, n_dvs), [&p, &range_evaluator](const range_t &range) {
            range_evaluator(p, range.begin(), range.end());
        });
    } else if (p.get_thread_safety() == thread_safety::basic) {
        // We cannot concurrently call the objfun on the input prob. We will need
        // to make a copy of p for each parallel iteration.
        tbb::parallel_for(range_t(0u, n_dvs), [p, &range_evaluator](const range_t &range) {
            range_evaluator(p, range.begin(), range.end());
        });
        // Manually increment the fitness eval counter in p. Since we used copies
        // of p for the parallel fitness evaluations, the counter in p did not change.
        p.increment_fevals(boost::numeric_cast<unsigned long long>(n_dvs));
    } else {
        pagmo_throw(std::invalid_argument, "Cannot use a thread_bfe on the problem '" + p.get_name()
                                               + "', which does not provide the required level of thread safety");
    }

    return retval;
}

// Serialization support.
template <typename Archive>
void thread_bfe::serialize(Archive &, unsigned)
{
}

} // namespace pagmo

PAGMO_S11N_BFE_IMPLEMENT(pagmo::thread_bfe)
