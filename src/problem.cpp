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
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

null_problem::null_problem(vector_double::size_type nobj, vector_double::size_type nec, vector_double::size_type nic,
                           vector_double::size_type nix)
    : m_nobj(nobj), m_nec(nec), m_nic(nic), m_nix(nix)
{
    if (!nobj) {
        pagmo_throw(std::invalid_argument, "The null problem must have a non-zero number of objectives");
    }
    if (nix > 1u) {
        pagmo_throw(std::invalid_argument, "The null problem must have an integer part strictly smaller than 2");
    }
}

/// Fitness.
/**
 * @return a zero-filled vector of size equal to the number of objectives.
 */
vector_double null_problem::fitness(const vector_double &) const
{
    return vector_double(get_nobj() + get_nec() + get_nic(), 0.);
}

/// Problem bounds.
/**
 * @return the pair <tt>([0.],[1.])</tt>.
 */
std::pair<vector_double, vector_double> null_problem::get_bounds() const
{
    return {{0.}, {1.}};
}

namespace detail
{

void check_problem_bounds(const std::pair<vector_double, vector_double> &bounds, vector_double::size_type nix)
{
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    // 0 - Check that the size is at least 1.
    if (lb.size() == 0u) {
        pagmo_throw(std::invalid_argument, "The bounds dimension cannot be zero");
    }
    // 1 - check bounds have equal length
    if (lb.size() != ub.size()) {
        pagmo_throw(std::invalid_argument, "The length of the lower bounds vector is " + std::to_string(lb.size())
                                               + ", the length of the upper bounds vector is "
                                               + std::to_string(ub.size()));
    }
    // 2 - checks lower < upper for all values in lb, ub, and check for nans.
    for (decltype(lb.size()) i = 0u; i < lb.size(); ++i) {
        if (std::isnan(lb[i]) || std::isnan(ub[i])) {
            pagmo_throw(std::invalid_argument,
                        "A NaN value was encountered in the problem bounds, index: " + std::to_string(i));
        }
        if (lb[i] > ub[i]) {
            pagmo_throw(std::invalid_argument,
                        "The lower bound at position " + std::to_string(i) + " is " + std::to_string(lb[i])
                            + " while the upper bound has the smaller value " + std::to_string(ub[i]));
        }
    }
    // 3 - checks the integer part
    if (nix) {
        const auto nx = lb.size();
        if (nix > nx) {
            pagmo_throw(std::invalid_argument, "The integer part cannot be larger than the bounds size");
        }
        const auto ncx = nx - nix;
        for (auto i = ncx; i < nx; ++i) {
            if (std::isfinite(lb[i]) && lb[i] != std::trunc(lb[i])) {
                pagmo_throw(std::invalid_argument, "A lower bound of the integer part of the decision vector is: "
                                                       + std::to_string(lb[i]) + " and is not an integer.");
            }
            if (std::isfinite(ub[i]) && ub[i] != std::trunc(ub[i])) {
                pagmo_throw(std::invalid_argument, "An upper bound of the integer part of the decision vector is: "
                                                       + std::to_string(ub[i]) + " and is not an integer.");
            }
        }
    }
}

// Helper functions to compute sparsity patterns in the dense case.
// A single dense hessian (lower triangular symmetric matrix).
sparsity_pattern dense_hessian(vector_double::size_type dim)
{
    sparsity_pattern retval;
    for (decltype(dim) j = 0u; j < dim; ++j) {
        for (decltype(dim) i = 0u; i <= j; ++i) {
            retval.emplace_back(j, i);
        }
    }
    return retval;
}

// A collection of f_dim identical dense hessians.
std::vector<sparsity_pattern> dense_hessians(vector_double::size_type f_dim, vector_double::size_type dim)
{
    return std::vector<sparsity_pattern>(boost::numeric_cast<std::vector<sparsity_pattern>::size_type>(f_dim),
                                         dense_hessian(dim));
}

// Dense gradient.
sparsity_pattern dense_gradient(vector_double::size_type f_dim, vector_double::size_type dim)
{
    sparsity_pattern retval;
    for (decltype(f_dim) j = 0u; j < f_dim; ++j) {
        for (decltype(dim) i = 0u; i < dim; ++i) {
            retval.emplace_back(j, i);
        }
    }
    return retval;
}

} // namespace detail

/// Default constructor.
/**
 * The default constructor will initialize a pagmo::problem containing a pagmo::null_problem.
 *
 * @throws unspecified any exception thrown by the constructor from UDP.
 */
problem::problem() : problem(null_problem{}) {}

// Implementation of the generic ctor.
void problem::generic_ctor_impl()
{
    // 0 - Integer part
    const auto tmp_size = ptr()->get_bounds().first.size();
    m_nix = ptr()->get_nix();
    if (m_nix > tmp_size) {
        pagmo_throw(std::invalid_argument, "The integer part of the problem (" + std::to_string(m_nix)
                                               + ") is larger than its dimension (" + std::to_string(tmp_size) + ")");
    }
    // 1 - Bounds.
    auto bounds = ptr()->get_bounds();
    detail::check_problem_bounds(bounds, m_nix);
    m_lb = std::move(bounds.first);
    m_ub = std::move(bounds.second);
    // 2 - Number of objectives.
    m_nobj = ptr()->get_nobj();
    if (!m_nobj) {
        pagmo_throw(std::invalid_argument, "The number of objectives cannot be zero");
    }
    // NOTE: here we check that we can always compute nobj + nec + nic safely.
    if (m_nobj > std::numeric_limits<decltype(m_nobj)>::max() / 3u) {
        pagmo_throw(std::invalid_argument, "The number of objectives is too large");
    }
    // 3 - Constraints.
    m_nec = ptr()->get_nec();
    if (m_nec > std::numeric_limits<decltype(m_nec)>::max() / 3u) {
        pagmo_throw(std::invalid_argument, "The number of equality constraints is too large");
    }
    m_nic = ptr()->get_nic();
    if (m_nic > std::numeric_limits<decltype(m_nic)>::max() / 3u) {
        pagmo_throw(std::invalid_argument, "The number of inequality constraints is too large");
    }
    // 4 - Presence of gradient and its sparsity.
    // NOTE: all these m_has_* attributes refer to the presence of the features in the UDP.
    m_has_gradient = ptr()->has_gradient();
    m_has_gradient_sparsity = ptr()->has_gradient_sparsity();
    // 5 - Presence of Hessians and their sparsity.
    m_has_hessians = ptr()->has_hessians();
    m_has_hessians_sparsity = ptr()->has_hessians_sparsity();
    // 5bis - Is this a stochastic problem?
    m_has_set_seed = ptr()->has_set_seed();
    // 6 - Name.
    m_name = ptr()->get_name();
    // 7 - Check the sparsities, and cache their sizes.
    if (m_has_gradient_sparsity) {
        // If the problem provides gradient sparsity, get it, check it
        // and store its size.
        const auto gs = ptr()->gradient_sparsity();
        check_gradient_sparsity(gs);
        m_gs_dim = gs.size();
    } else {
        // If the problem does not provide gradient sparsity, we assume dense
        // sparsity. We can compute easily the expected size of the sparsity
        // in this case.
        const auto nx = get_nx();
        const auto nf = get_nf();
        if (nx > std::numeric_limits<vector_double::size_type>::max() / nf) {
            pagmo_throw(std::invalid_argument, "The size of the (dense) gradient sparsity is too large");
        }
        m_gs_dim = nx * nf;
    }
    // Same as above for the hessians.
    if (m_has_hessians_sparsity) {
        const auto hs = ptr()->hessians_sparsity();
        check_hessians_sparsity(hs);
        for (const auto &one_hs : hs) {
            m_hs_dim.push_back(one_hs.size());
        }
    } else {
        const auto nx = get_nx();
        const auto nf = get_nf();
        if (nx == std::numeric_limits<vector_double::size_type>::max()
            || nx / 2u > std::numeric_limits<vector_double::size_type>::max() / (nx + 1u)) {
            pagmo_throw(std::invalid_argument, "The size of the (dense) hessians sparsity is too large");
        }
        // We resize rather than push back here, so that an std::length_error is called quickly rather
        // than an std::bad_alloc after waiting the growth
        m_hs_dim.resize(boost::numeric_cast<decltype(m_hs_dim.size())>(nf));
        std::fill(m_hs_dim.begin(), m_hs_dim.end(), nx * (nx - 1u) / 2u + nx); // lower triangular
    }
    // 8 - Constraint tolerance
    m_c_tol.resize(m_nec + m_nic);
    // 9 - Thread safety.
    m_thread_safety = ptr()->get_thread_safety();
}

} // namespace pagmo
