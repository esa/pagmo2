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
#include <pagmo/r_policies/fair_replace.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/s11n.hpp>
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

// Default constructor: absolute rate, 1 individual.
fair_replace::fair_replace() : fair_replace(1) {}

// Implementation of the replacement.
individuals_group_t fair_replace::replace(const individuals_group_t &inds, const vector_double::size_type &,
                                          const vector_double::size_type &, const vector_double::size_type &nobj,
                                          const vector_double::size_type &nec, const vector_double::size_type &nic,
                                          const vector_double &tol, const individuals_group_t &mig) const
{
    if (nobj > 1u && (nic || nec)) {
        pagmo_throw(std::invalid_argument, "The 'fair_replace' replacement policy is unable to deal with "
                                           "multiobjective constrained optimisation problems");
    }

    // Cache the sizes of the input pop and the migrants.
    // NOTE: use the size type of the dvs, which is pop_size_t.
    const auto inds_size = std::get<1>(inds).size();
    const auto mig_size = std::get<1>(mig).size();

    // Establish how many individuals we want to migrate from mig into inds.
    const auto n_migr = [this, inds_size, mig_size]() -> pop_size_t {
        pop_size_t candidate;

        if (this->m_migr_rate.which()) {
            // Fractional migration rate: scale it by the number
            // of input individuals.
            // NOTE: use std::min() to make absolutely sure we don't exceed inds_size
            // due to FP shenanigans.
            candidate = std::min(
                boost::numeric_cast<pop_size_t>(boost::get<double>(m_migr_rate) * static_cast<double>(inds_size)),
                inds_size);
        } else {
            // Absolute migration rate: check that it's not higher than the input population size.
            candidate = boost::get<pop_size_t>(m_migr_rate);
            if (candidate > inds_size) {
                pagmo_throw(
                    std::invalid_argument,
                    "The absolute migration rate (" + std::to_string(candidate)
                        + ") in a 'fair_replace' replacement policy is larger than the number of input individuals ("
                        + std::to_string(inds_size) + ")");
            }
        }

        // We cannot migrate more individuals than we have available
        // in mig, so clamp the candidate value.
        return std::min(candidate, mig_size);
    }();

    // Make extra sure that the number of individuals selected
    // for migration is not larger than mig_size.
    assert(n_migr <= mig_size);

    // NOTE: currently this replacement policy can handle:
    // - single-ojective (un)constrained optimisation,
    // - multiobjective unconstrained optimisation.
    // We already checked above that we are not in an MO
    // constrained case.
    if (nobj == 1u && !nic && !nec) {
        // Single-objective, unconstrained.

        // Sort (indirectly) the migrants according to their fitness.
        std::vector<pop_size_t> mig_ind_sort;
        mig_ind_sort.resize(boost::numeric_cast<decltype(mig_ind_sort.size())>(mig_size));
        std::iota(mig_ind_sort.begin(), mig_ind_sort.end(), pop_size_t(0));
        std::sort(mig_ind_sort.begin(), mig_ind_sort.end(), [&mig](pop_size_t idx1, pop_size_t idx2) {
            return detail::less_than_f(std::get<2>(mig)[idx1][0], std::get<2>(mig)[idx2][0]);
        });

        // Build the merged population from the original individuals plus the
        // top n_migr migrants.
        auto merged_pop(inds);
        for (pop_size_t i = 0; i < n_migr; ++i) {
            std::get<0>(merged_pop).push_back(std::get<0>(mig)[mig_ind_sort[i]]);
            std::get<1>(merged_pop).push_back(std::get<1>(mig)[mig_ind_sort[i]]);
            std::get<2>(merged_pop).push_back(std::get<2>(mig)[mig_ind_sort[i]]);
        }

        // Sort (indirectly) the merged population.
        std::vector<pop_size_t> merged_pop_ind_sort;
        merged_pop_ind_sort.resize(
            boost::numeric_cast<decltype(merged_pop_ind_sort.size())>(std::get<0>(merged_pop).size()));
        std::iota(merged_pop_ind_sort.begin(), merged_pop_ind_sort.end(), pop_size_t(0));
        std::sort(merged_pop_ind_sort.begin(), merged_pop_ind_sort.end(),
                  [&merged_pop](pop_size_t idx1, pop_size_t idx2) {
                      return detail::less_than_f(std::get<2>(merged_pop)[idx1][0], std::get<2>(merged_pop)[idx2][0]);
                  });

        // Create and return the output pop.
        individuals_group_t retval;
        std::get<0>(retval).reserve(std::get<0>(inds).size());
        std::get<1>(retval).reserve(std::get<1>(inds).size());
        std::get<2>(retval).reserve(std::get<2>(inds).size());
        for (pop_size_t i = 0; i < inds_size; ++i) {
            std::get<0>(retval).push_back(std::get<0>(merged_pop)[merged_pop_ind_sort[i]]);
            std::get<1>(retval).push_back(std::get<1>(merged_pop)[merged_pop_ind_sort[i]]);
            std::get<2>(retval).push_back(std::get<2>(merged_pop)[merged_pop_ind_sort[i]]);
        }

        return retval;
    } else if (nobj == 1u && (nic || nec)) {
        // Single-objective, constrained.

        // Sort indirectly the input migrants, taking into accounts
        // constraints satisfaction and tolerances.
        const auto mig_ind_sort = sort_population_con(std::get<2>(mig), nec, tol);

        // Build the merged population from the original individuals plus the
        // top n_migr migrants.
        auto merged_pop(inds);
        for (pop_size_t i = 0; i < n_migr; ++i) {
            std::get<0>(merged_pop).push_back(std::get<0>(mig)[mig_ind_sort[i]]);
            std::get<1>(merged_pop).push_back(std::get<1>(mig)[mig_ind_sort[i]]);
            std::get<2>(merged_pop).push_back(std::get<2>(mig)[mig_ind_sort[i]]);
        }

        // Sort indirectly the merged population.
        const auto merged_pop_ind_sort = sort_population_con(std::get<2>(merged_pop), nec, tol);

        // Create and return the output pop.
        individuals_group_t retval;
        std::get<0>(retval).reserve(std::get<0>(inds).size());
        std::get<1>(retval).reserve(std::get<1>(inds).size());
        std::get<2>(retval).reserve(std::get<2>(inds).size());
        for (pop_size_t i = 0; i < inds_size; ++i) {
            std::get<0>(retval).push_back(std::get<0>(merged_pop)[merged_pop_ind_sort[i]]);
            std::get<1>(retval).push_back(std::get<1>(merged_pop)[merged_pop_ind_sort[i]]);
            std::get<2>(retval).push_back(std::get<2>(merged_pop)[merged_pop_ind_sort[i]]);
        }

        return retval;
    } else {
        // Multi-objective, unconstrained.
        assert(nobj > 1u && !nic && !nec);

        // Get the best n_migr migrants.
        const auto mig_ind_sort = select_best_N_mo(std::get<2>(mig), n_migr);

        // Build the merged population from the original individuals plus the
        // top n_migr migrants.
        auto merged_pop(inds);
        for (pop_size_t i = 0; i < n_migr; ++i) {
            std::get<0>(merged_pop).push_back(std::get<0>(mig)[mig_ind_sort[i]]);
            std::get<1>(merged_pop).push_back(std::get<1>(mig)[mig_ind_sort[i]]);
            std::get<2>(merged_pop).push_back(std::get<2>(mig)[mig_ind_sort[i]]);
        }

        // Get the best inds_size individuals from the merged population.
        const auto merged_pop_ind_sort = select_best_N_mo(std::get<2>(merged_pop), inds_size);

        // Create and return the output pop.
        individuals_group_t retval;
        std::get<0>(retval).reserve(std::get<0>(inds).size());
        std::get<1>(retval).reserve(std::get<1>(inds).size());
        std::get<2>(retval).reserve(std::get<2>(inds).size());
        for (pop_size_t i = 0; i < inds_size; ++i) {
            std::get<0>(retval).push_back(std::get<0>(merged_pop)[merged_pop_ind_sort[i]]);
            std::get<1>(retval).push_back(std::get<1>(merged_pop)[merged_pop_ind_sort[i]]);
            std::get<2>(retval).push_back(std::get<2>(merged_pop)[merged_pop_ind_sort[i]]);
        }

        return retval;
    }
}

// Extra info.
std::string fair_replace::get_extra_info() const
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
void fair_replace::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, boost::serialization::base_object<detail::base_sr_policy>(*this));
}

} // namespace pagmo

PAGMO_S11N_R_POLICY_IMPLEMENT(pagmo::fair_replace)
