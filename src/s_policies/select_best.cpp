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
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/variant/get.hpp>

#include <pagmo/detail/base_sr_policy.hpp>
#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/s_policies/select_best.hpp>
#include <pagmo/s_policy.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>
#include <pagmo/utils/multi_objective.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

// Default constructor: absolute migration rate, 1 individual.
select_best::select_best() : select_best(1) {}

// Implementation of the selection.
individuals_group_t select_best::select(const individuals_group_t &inds, const vector_double::size_type &,
                                        const vector_double::size_type &, const vector_double::size_type &nobj,
                                        const vector_double::size_type &nec, const vector_double::size_type &nic,
                                        const vector_double &tol) const
{
    if (nobj > 1u && (nic || nec)) {
        pagmo_throw(std::invalid_argument, "The 'Select best' selection policy is unable to deal with "
                                           "multiobjective constrained optimisation problems");
    }

    // Cache the sizes of the input pop.
    // NOTE: use the size type of the dvs, which is pop_size_t.
    const auto inds_size = std::get<1>(inds).size();

    // Establish how many individuals we want to select from inds.
    const auto n_migr = [this, inds_size]() -> pop_size_t {
        if (this->m_migr_rate.which()) {
            // Fractional migration rate: scale it by the number
            // of input individuals.
            // NOTE: use std::min() to make absolutely sure we don't exceed inds_size
            // due to FP shenanigans.
            return std::min(
                boost::numeric_cast<pop_size_t>(boost::get<double>(m_migr_rate) * static_cast<double>(inds_size)),
                inds_size);
        } else {
            // Absolute migration rate: check that it's not higher than the input population size.
            const auto candidate = boost::get<pop_size_t>(m_migr_rate);
            if (candidate > inds_size) {
                pagmo_throw(
                    std::invalid_argument,
                    "The absolute migration rate (" + std::to_string(candidate)
                        + ") in a 'Select best' selection policy is larger than the number of input individuals ("
                        + std::to_string(inds_size) + ")");
            }
            return candidate;
        }
    }();

    // Make extra sure that the number of individuals selected
    // is not larger than inds_size.
    assert(n_migr <= inds_size);

    // NOTE: currently this selection policy can handle:
    // - single-ojective (un)constrained optimisation,
    // - multiobjective unconstrained optimisation.
    // We already checked above that we are not in an MO
    // constrained case.
    if (nobj == 1u && !nic && !nec) {
        // Single-objective, unconstrained.

        // Sort (indirectly) the input individuals according to their fitness.
        std::vector<pop_size_t> inds_ind_sort;
        inds_ind_sort.resize(boost::numeric_cast<decltype(inds_ind_sort.size())>(inds_size));
        std::iota(inds_ind_sort.begin(), inds_ind_sort.end(), pop_size_t(0));
        std::sort(inds_ind_sort.begin(), inds_ind_sort.end(), [&inds](pop_size_t idx1, pop_size_t idx2) {
            return detail::less_than_f(std::get<2>(inds)[idx1][0], std::get<2>(inds)[idx2][0]);
        });

        // Create and return the output pop.
        individuals_group_t retval;
        std::get<0>(retval).reserve(n_migr);
        std::get<1>(retval).reserve(n_migr);
        std::get<2>(retval).reserve(n_migr);
        for (pop_size_t i = 0; i < n_migr; ++i) {
            std::get<0>(retval).push_back(std::get<0>(inds)[inds_ind_sort[i]]);
            std::get<1>(retval).push_back(std::get<1>(inds)[inds_ind_sort[i]]);
            std::get<2>(retval).push_back(std::get<2>(inds)[inds_ind_sort[i]]);
        }

        return retval;
    } else if (nobj == 1u && (nic || nec)) {
        // Single-objective, constrained.

        // Sort indirectly the input individuals, taking into accounts
        // constraints satisfaction and tolerances.
        const auto inds_ind_sort = sort_population_con(std::get<2>(inds), nec, tol);

        // Create and return the output pop.
        individuals_group_t retval;
        std::get<0>(retval).reserve(n_migr);
        std::get<1>(retval).reserve(n_migr);
        std::get<2>(retval).reserve(n_migr);
        for (pop_size_t i = 0; i < n_migr; ++i) {
            std::get<0>(retval).push_back(std::get<0>(inds)[inds_ind_sort[i]]);
            std::get<1>(retval).push_back(std::get<1>(inds)[inds_ind_sort[i]]);
            std::get<2>(retval).push_back(std::get<2>(inds)[inds_ind_sort[i]]);
        }

        return retval;
    } else {
        // Multi-objective, unconstrained.
        assert(nobj > 1u && !nic && !nec);

        // Get the best n_migr individuals.
        const auto inds_ind_sort = select_best_N_mo(std::get<2>(inds), n_migr);

        // Create and return the output pop.
        individuals_group_t retval;
        std::get<0>(retval).reserve(n_migr);
        std::get<1>(retval).reserve(n_migr);
        std::get<2>(retval).reserve(n_migr);
        for (pop_size_t i = 0; i < n_migr; ++i) {
            std::get<0>(retval).push_back(std::get<0>(inds)[inds_ind_sort[i]]);
            std::get<1>(retval).push_back(std::get<1>(inds)[inds_ind_sort[i]]);
            std::get<2>(retval).push_back(std::get<2>(inds)[inds_ind_sort[i]]);
        }

        return retval;
    }
}

// Extra info.
std::string select_best::get_extra_info() const
{
    if (m_migr_rate.which()) {
        const auto rate = boost::get<double>(m_migr_rate);
        return "\tFractional migration rate: " + std::to_string(rate);
    } else {
        const auto rate = boost::get<pop_size_t>(m_migr_rate);
        return "\tAbsolute migration rate: " + std::to_string(rate);
    }
}

// Serialization support.
template <typename Archive>
void select_best::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, boost::serialization::base_object<detail::base_sr_policy>(*this));
}

} // namespace pagmo

PAGMO_S11N_S_POLICY_IMPLEMENT(pagmo::select_best)
