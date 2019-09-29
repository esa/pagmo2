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
#include <atomic>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pagmo/detail/bfe_impl.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/null_problem.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

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
    if (m_nobj > std::numeric_limits<vector_double::size_type>::max() / 3u) {
        pagmo_throw(std::invalid_argument, "The number of objectives is too large");
    }
    // 3 - Constraints.
    m_nec = ptr()->get_nec();
    if (m_nec > std::numeric_limits<vector_double::size_type>::max() / 3u) {
        pagmo_throw(std::invalid_argument, "The number of equality constraints is too large");
    }
    m_nic = ptr()->get_nic();
    if (m_nic > std::numeric_limits<vector_double::size_type>::max() / 3u) {
        pagmo_throw(std::invalid_argument, "The number of inequality constraints is too large");
    }
    // 4 - Presence of batch_fitness().
    // NOTE: all these m_has_* attributes refer to the presence of the features in the UDP.
    m_has_batch_fitness = ptr()->has_batch_fitness();
    // 5 - Presence of gradient and its sparsity.
    m_has_gradient = ptr()->has_gradient();
    m_has_gradient_sparsity = ptr()->has_gradient_sparsity();
    // 6 - Presence of Hessians and their sparsity.
    m_has_hessians = ptr()->has_hessians();
    m_has_hessians_sparsity = ptr()->has_hessians_sparsity();
    // 7 - Is this a stochastic problem?
    m_has_set_seed = ptr()->has_set_seed();
    // 8 - Name.
    m_name = ptr()->get_name();
    // 9 - Check the sparsities, and cache their sizes.
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
    // 10 - Constraint tolerance
    m_c_tol.resize(m_nec + m_nic);
    // 11 - Thread safety.
    m_thread_safety = ptr()->get_thread_safety();
}

/// Copy constructor.
/**
 * The copy constructor will deep copy the input problem \p other.
 *
 * @param other the problem to be copied.
 *
 * @throws unspecified any exception thrown by:
 * - memory allocation errors in standard containers,
 * - the copying of the internal UDP.
 */
problem::problem(const problem &other)
    : m_ptr(other.ptr()->clone()), m_fevals(other.m_fevals.load(std::memory_order_relaxed)),
      m_gevals(other.m_gevals.load(std::memory_order_relaxed)),
      m_hevals(other.m_hevals.load(std::memory_order_relaxed)), m_lb(other.m_lb), m_ub(other.m_ub),
      m_nobj(other.m_nobj), m_nec(other.m_nec), m_nic(other.m_nic), m_nix(other.m_nix), m_c_tol(other.m_c_tol),
      m_has_batch_fitness(other.m_has_batch_fitness), m_has_gradient(other.m_has_gradient),
      m_has_gradient_sparsity(other.m_has_gradient_sparsity), m_has_hessians(other.m_has_hessians),
      m_has_hessians_sparsity(other.m_has_hessians_sparsity), m_has_set_seed(other.m_has_set_seed),
      m_name(other.m_name), m_gs_dim(other.m_gs_dim), m_hs_dim(other.m_hs_dim), m_thread_safety(other.m_thread_safety)
{
}

/// Move constructor.
/**
 * @param other the problem from which \p this will be move-constructed.
 */
problem::problem(problem &&other) noexcept
    : m_ptr(std::move(other.m_ptr)), m_fevals(other.m_fevals.load(std::memory_order_relaxed)),
      m_gevals(other.m_gevals.load(std::memory_order_relaxed)),
      m_hevals(other.m_hevals.load(std::memory_order_relaxed)), m_lb(std::move(other.m_lb)),
      m_ub(std::move(other.m_ub)), m_nobj(other.m_nobj), m_nec(other.m_nec), m_nic(other.m_nic), m_nix(other.m_nix),
      m_c_tol(std::move(other.m_c_tol)), m_has_batch_fitness(other.m_has_batch_fitness),
      m_has_gradient(other.m_has_gradient), m_has_gradient_sparsity(other.m_has_gradient_sparsity),
      m_has_hessians(other.m_has_hessians), m_has_hessians_sparsity(other.m_has_hessians_sparsity),
      m_has_set_seed(other.m_has_set_seed), m_name(std::move(other.m_name)), m_gs_dim(other.m_gs_dim),
      m_hs_dim(other.m_hs_dim), m_thread_safety(std::move(other.m_thread_safety))
{
}

/// Move assignment operator
/**
 * @param other the assignment target.
 *
 * @return a reference to \p this.
 */
problem &problem::operator=(problem &&other) noexcept
{
    if (this != &other) {
        m_ptr = std::move(other.m_ptr);
        m_fevals.store(other.m_fevals.load(std::memory_order_relaxed), std::memory_order_relaxed);
        m_gevals.store(other.m_gevals.load(std::memory_order_relaxed), std::memory_order_relaxed);
        m_hevals.store(other.m_hevals.load(std::memory_order_relaxed), std::memory_order_relaxed);
        m_lb = std::move(other.m_lb);
        m_ub = std::move(other.m_ub);
        m_nobj = other.m_nobj;
        m_nec = other.m_nec;
        m_nic = other.m_nic;
        m_nix = other.m_nix;
        m_c_tol = std::move(other.m_c_tol);
        m_has_batch_fitness = other.m_has_batch_fitness;
        m_has_gradient = other.m_has_gradient;
        m_has_gradient_sparsity = other.m_has_gradient_sparsity;
        m_has_hessians = other.m_has_hessians;
        m_has_hessians_sparsity = other.m_has_hessians_sparsity;
        m_has_set_seed = other.m_has_set_seed;
        m_name = std::move(other.m_name);
        m_gs_dim = other.m_gs_dim;
        m_hs_dim = std::move(other.m_hs_dim);
        m_thread_safety = std::move(other.m_thread_safety);
    }
    return *this;
}

/// Copy assignment operator
/**
 * Copy assignment is implemented as a copy constructor followed by a move assignment.
 *
 * @param other the assignment target.
 *
 * @return a reference to \p this.
 *
 * @throws unspecified any exception thrown by the copy constructor.
 */
problem &problem::operator=(const problem &other)
{
    // Copy ctor + move assignment.
    return *this = problem(other);
}

/// Fitness.
/**
 * This method will invoke the <tt>%fitness()</tt> method of the UDP to compute the fitness of the
 * input decision vector \p dv. The return value of the <tt>%fitness()</tt> method of the UDP is expected to have a
 * dimension of \f$n_{f} = n_{obj} + n_{ec} + n_{ic}\f$
 * and to contain the concatenated values of \f$\mathbf f, \mathbf c_e\f$ and \f$\mathbf c_i\f$ (in this order).
 * Equality constraints are all assumed in the form \f$c_{e_i}(\mathbf x) = 0\f$ while inequalities are assumed in
 * the form \f$c_{i_i}(\mathbf x) <= 0\f$ so that negative values are associated to satisfied inequalities.
 *
 * In addition to invoking the <tt>%fitness()</tt> method of the UDP, this method will perform sanity checks on
 * \p dv and on the returned fitness vector. A successful call of this method will increase the internal fitness
 * evaluation counter (see problem::get_fevals()).
 *
 * @param dv the decision vector.
 *
 * @return the fitness of \p dv.
 *
 * @throws std::invalid_argument if either
 * - the length of \p dv differs from the value returned by get_nx(), or
 * - the length of the returned fitness vector differs from the the value returned by get_nf().
 * @throws unspecified any exception thrown by the <tt>%fitness()</tt> method of the UDP.
 */
vector_double problem::fitness(const vector_double &dv) const
{
    // NOTE: it is important that we can call this method on the same object from multiple threads,
    // for parallel initialisation in populations/archis. Such thread safety must be maintained
    // if we change the implementation of this method.

    // 1 - checks the decision vector
    // NOTE: the check uses UDP properties cached on construction. This is const and thread-safe.
    detail::prob_check_dv(*this, dv.data(), dv.size());

    // 2 - computes the fitness
    // NOTE: the thread safety here depends on the thread safety of the UDP. We make sure in the
    // parallel init methods that we never invoke this method concurrently if the UDP is not
    // sufficiently thread-safe.
    vector_double retval(ptr()->fitness(dv));

    // 3 - checks the fitness vector
    // NOTE: as above, we are just making sure the fitness length is consistent with the fitness
    // length stored in the problem. This is const and thread-safe.
    detail::prob_check_fv(*this, retval.data(), retval.size());

    // 4 - increments fitness evaluation counter
    // NOTE: this is an atomic variable, thread-safe.
    increment_fevals(1);

    return retval;
}

/// Batch fitness.
/**
 * This method implements the evaluation of multiple decision vectors in batch mode
 * by invoking the <tt>%batch_fitness()</tt> method of the UDP. The <tt>%batch_fitness()</tt>
 * method of the UDP accepts in input a batch of decision vectors, \p dvs, stored contiguously:
 * for a problem with dimension \f$ n \f$, the first decision vector in \p dvs occupies
 * the index range \f$ \left[0, n\right) \f$, the second decision vector occupies the range
 * \f$ \left[n, 2n\right) \f$, and so on. The return value is the batch of fitness vectors \p fvs
 * resulting from computing the fitness of the input decision vectors.
 * \p fvs is also stored contiguously: for a problem with fitness dimension \f$ f \f$, the first fitness
 * vector will occupy the index range \f$ \left[0, f\right) \f$, the second fitness vector
 * will occupy the range \f$ \left[f, 2f\right) \f$, and so on.
 *
 * \verbatim embed:rst:leading-asterisk
 * If the UDP satisfies :cpp:class:`pagmo::has_batch_fitness`, this method will forward ``dvs``
 * to the ``batch_fitness()`` method of the UDP after sanity checks. The output of the ``batch_fitness()``
 * method of the UDP will also be checked before being returned. If the UDP does not satisfy
 * :cpp:class:`pagmo::has_batch_fitness`, an error will be raised.
 * \endverbatim
 *
 * A successful call of this method will increase the internal fitness evaluation counter (see
 * problem::get_fevals()).
 *
 * @param dvs the input batch of decision vectors.
 *
 * @return the fitnesses of the decision vectors in \p dvs.
 *
 * @throws std::invalid_argument if either \p dvs or the return value are incompatible
 * with the properties of this problem.
 * @throws not_implemented_error if the UDP does not have a <tt>%batch_fitness()</tt> method.
 * @throws unspecified any exception thrown by the <tt>%batch_fitness()</tt> method of the UDP.
 */
vector_double problem::batch_fitness(const vector_double &dvs) const
{
    // Check the input dvs.
    detail::bfe_check_input_dvs(*this, dvs);

    // Invoke the batch fitness function of the UDP, and
    // increase the fevals counter as well.
    auto retval = detail::prob_invoke_mem_batch_fitness(*this, dvs);

    // Check the produced vector of fitnesses.
    detail::bfe_check_output_fvs(*this, dvs, retval);

    return retval;
}

/// Gradient.
/**
 * This method will compute the gradient of the input decision vector \p dv by invoking
 * the <tt>%gradient()</tt> method of the UDP. The <tt>%gradient()</tt> method of the UDP must return
 * a sparse representation of the gradient: the \f$ k\f$-th term of the gradient vector
 * is expected to contain \f$ \frac{\partial f_i}{\partial x_j}\f$, where the pair \f$(i,j)\f$
 * is the \f$k\f$-th element of the sparsity pattern (collection of index pairs), as returned by
 * problem::gradient_sparsity().
 *
 * If the UDP satisfies pagmo::has_gradient, this method will forward \p dv to the <tt>%gradient()</tt>
 * method of the UDP after sanity checks. The output of the <tt>%gradient()</tt>
 * method of the UDP will also be checked before being returned. If the UDP does not satisfy
 * pagmo::has_gradient, an error will be raised.
 *
 * A successful call of this method will increase the internal gradient evaluation counter (see
 * problem::get_gevals()).
 *
 * @param dv the decision vector whose gradient will be computed.
 *
 * @return the gradient of \p dv.
 *
 * @throws std::invalid_argument if either:
 * - the length of \p dv differs from the value returned by get_nx(), or
 * - the returned gradient vector does not have the same size as the vector returned by
 *   problem::gradient_sparsity().
 * @throws not_implemented_error if the UDP does not satisfy pagmo::has_gradient.
 * @throws unspecified any exception thrown by the <tt>%gradient()</tt> method of the UDP.
 */
vector_double problem::gradient(const vector_double &dv) const
{
    // 1 - checks the decision vector
    detail::prob_check_dv(*this, dv.data(), dv.size());
    // 2 - compute the gradients
    vector_double retval(ptr()->gradient(dv));
    // 3 - checks the gradient vector
    check_gradient_vector(retval);
    // 4 - increments gradient evaluation counter
    m_gevals.fetch_add(1u, std::memory_order_relaxed);
    return retval;
}

/// Gradient sparsity pattern.
/**
 * This method will return the gradient sparsity pattern of the problem. The gradient sparsity pattern is a
 * lexicographically sorted collection of the indices \f$(i,j)\f$ of the non-zero elements of
 * \f$ g_{ij} = \frac{\partial f_i}{\partial x_j}\f$.
 *
 * If problem::has_gradient_sparsity() returns \p true,
 * then the <tt>%gradient_sparsity()</tt> method of the UDP will be invoked, and its result returned (after sanity
 * checks). Otherwise, a a dense pattern is assumed and the returned vector will be
 * \f$((0,0),(0,1), ... (0,n_x-1), ...(n_f-1,n_x-1))\f$.
 *
 * @return the gradient sparsity pattern.
 *
 * @throws std::invalid_argument if the sparsity pattern returned by the UDP is invalid (specifically, if
 * it is not strictly sorted lexicographically, or if the indices in the pattern are incompatible with the
 * properties of the problem, or if the size of the returned pattern is different from the size recorded upon
 * construction).
 * @throws unspecified memory errors in standard containers.
 */
sparsity_pattern problem::gradient_sparsity() const
{
    if (has_gradient_sparsity()) {
        auto retval = ptr()->gradient_sparsity();
        check_gradient_sparsity(retval);
        // Check the size is consistent with the stored size.
        // NOTE: we need to do this check here, and not in check_gradient_sparsity(),
        // because check_gradient_sparsity() is sometimes called when m_gs_dim has not been
        // initialised yet (e.g., in the ctor).
        if (retval.size() != m_gs_dim) {
            pagmo_throw(std::invalid_argument,
                        "Invalid gradient sparsity pattern: the returned sparsity pattern has a size of "
                            + std::to_string(retval.size())
                            + ", while the sparsity pattern size stored upon problem construction is "
                            + std::to_string(m_gs_dim));
        }
        return retval;
    }
    return detail::dense_gradient(get_nf(), get_nx());
}

/// Hessians.
/**
 * This method will compute the hessians of the input decision vector \p dv by invoking
 * the <tt>%hessians()</tt> method of the UDP. The <tt>%hessians()</tt> method of the UDP must return
 * a sparse representation of the hessians: the element \f$ l\f$ of the returned vector contains
 * \f$ h^l_{ij} = \frac{\partial f^2_l}{\partial x_i\partial x_j}\f$
 * in the order specified by the \f$ l\f$-th element of the
 * hessians sparsity pattern (a vector of index pairs \f$(i,j)\f$)
 * as returned by problem::hessians_sparsity(). Since
 * the hessians are symmetric, their sparse representation contains only lower triangular elements.
 *
 * If the UDP satisfies pagmo::has_hessians, this method will forward \p dv to the <tt>%hessians()</tt>
 * method of the UDP after sanity checks. The output of the <tt>%hessians()</tt>
 * method of the UDP will also be checked before being returned. If the UDP does not satisfy
 * pagmo::has_hessians, an error will be raised.
 *
 * A successful call of this method will increase the internal hessians evaluation counter (see
 * problem::get_hevals()).
 *
 * @param dv the decision vector whose hessians will be computed.
 *
 * @return the hessians of \p dv.
 *
 * @throws std::invalid_argument if either:
 * - the length of \p dv differs from the output of get_nx(), or
 * - the length of the returned hessians does not match the corresponding hessians sparsity pattern dimensions, or
 * - the size of the return value is not equal to the fitness dimension.
 * @throws not_implemented_error if the UDP does not satisfy pagmo::has_hessians.
 * @throws unspecified any exception thrown by the <tt>%hessians()</tt> method of the UDP.
 */
std::vector<vector_double> problem::hessians(const vector_double &dv) const
{
    // 1 - checks the decision vector
    detail::prob_check_dv(*this, dv.data(), dv.size());
    // 2 - computes the hessians
    auto retval(ptr()->hessians(dv));
    // 3 - checks the hessians
    check_hessians_vector(retval);
    // 4 - increments hessians evaluation counter
    m_hevals.fetch_add(1u, std::memory_order_relaxed);
    return retval;
}

/// Hessians sparsity pattern.
/**
 * This method will return the hessians sparsity pattern of the problem. Each component \f$ l\f$ of the hessians
 * sparsity pattern is a lexicographically sorted collection of the indices \f$(i,j)\f$ of the non-zero elements of
 * \f$h^l_{ij} = \frac{\partial f^l}{\partial x_i\partial x_j}\f$. Since the Hessian matrix
 * is symmetric, only lower triangular elements are allowed.
 *
 * If problem::has_hessians_sparsity() returns \p true,
 * then the <tt>%hessians_sparsity()</tt> method of the UDP will be invoked, and its result returned (after sanity
 * checks). Otherwise, a dense pattern is assumed and \f$n_f\f$ sparsity patterns
 * containing \f$((0,0),(1,0), (1,1), (2,0) ... (n_x-1,n_x-1))\f$ will be returned.
 *
 * @return the hessians sparsity pattern.
 *
 * @throws std::invalid_argument if a sparsity pattern returned by the UDP is invalid (specifically, if
 * if it is not strictly sorted lexicographically, if the returned indices do not
 * correspond to a lower triangular representation of a symmetric matrix, or if the size of the pattern differs
 * from the size recorded upon construction).
 */
std::vector<sparsity_pattern> problem::hessians_sparsity() const
{
    if (m_has_hessians_sparsity) {
        auto retval = ptr()->hessians_sparsity();
        check_hessians_sparsity(retval);
        // Check the sizes are consistent with the stored sizes.
        // NOTE: we need to do this check here, and not in check_hessians_sparsity(),
        // because check_hessians_sparsity() is sometimes called when m_hs_dim has not been
        // initialised yet (e.g., in the ctor).
        // NOTE: in check_hessians_sparsity() we have already checked the size of retval. It has
        // to be the same as the fitness dimension. The same check is run when m_hs_dim is originally
        // created, hence they must be equal.
        assert(retval.size() == m_hs_dim.size());
        auto r_it = retval.begin();
        for (const auto &dim : m_hs_dim) {
            if (r_it->size() != dim) {
                pagmo_throw(std::invalid_argument,
                            "Invalid hessian sparsity pattern: the returned sparsity pattern has a size of "
                                + std::to_string(r_it->size())
                                + ", while the sparsity pattern size stored upon problem construction is "
                                + std::to_string(dim));
            }
            ++r_it;
        }
        return retval;
    }
    return detail::dense_hessians(get_nf(), get_nx());
}

/// Box-bounds.
/**
 * @return \f$ (\mathbf{lb}, \mathbf{ub}) \f$, the box-bounds, as returned by
 * the <tt>%get_bounds()</tt> method of the UDP. Infinities in the bounds are allowed.
 *
 * @throws unspecified any exception thrown by memory errors in standard containers.
 */
std::pair<vector_double, vector_double> problem::get_bounds() const
{
    return std::make_pair(m_lb, m_ub);
}

/// Set the constraint tolerance (from a vector of doubles).
/**
 * @param c_tol a vector containing the tolerances to use when
 * checking for constraint feasibility.
 *
 * @throws std::invalid_argument if the size of \p c_tol differs from the number of constraints, or if
 * any of its elements is negative or NaN.
 */
void problem::set_c_tol(const vector_double &c_tol)
{
    if (c_tol.size() != this->get_nc()) {
        pagmo_throw(std::invalid_argument, "The tolerance vector size should be: " + std::to_string(this->get_nc())
                                               + ", while a size of: " + std::to_string(c_tol.size())
                                               + " was detected.");
    }
    for (decltype(c_tol.size()) i = 0; i < c_tol.size(); ++i) {
        if (std::isnan(c_tol[i])) {
            pagmo_throw(std::invalid_argument,
                        "The tolerance vector has a NaN value at the index " + std::to_string(i));
        }
        if (c_tol[i] < 0.) {
            pagmo_throw(std::invalid_argument,
                        "The tolerance vector has a negative value at the index " + std::to_string(i));
        }
    }
    m_c_tol = c_tol;
}

/// Set the constraint tolerance (from a single double value).
/**
 * @param c_tol the tolerance to use when checking for all constraint feasibilities.
 *
 * @throws std::invalid_argument if \p c_tol is negative or NaN.
 */
void problem::set_c_tol(double c_tol)
{
    if (std::isnan(c_tol)) {
        pagmo_throw(std::invalid_argument, "The tolerance cannot be set to be NaN.");
    }
    if (c_tol < 0.) {
        pagmo_throw(std::invalid_argument, "The tolerance cannot be negative.");
    }
    m_c_tol = vector_double(this->get_nc(), c_tol);
}

/// Set the seed for the stochastic variables.
/**
 * Sets the seed to be used in the fitness function to instantiate
 * all stochastic variables. If the UDP satisfies pagmo::has_set_seed, then
 * its <tt>%set_seed()</tt> method will be invoked. Otherwise, an error will be raised.
 *
 * @param seed seed.
 *
 * @throws not_implemented_error if the UDP does not satisfy pagmo::has_set_seed.
 * @throws unspecified any exception thrown by the <tt>%set_seed()</tt> method of the UDP.
 */
void problem::set_seed(unsigned seed)
{
    ptr()->set_seed(seed);
}

/// Feasibility of a decision vector.
/**
 * This method will check the feasibility of the fitness corresponding to
 * a decision vector \p x against
 * the tolerances returned by problem::get_c_tol().
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    One call of this method will cause one call to the fitness function.
 *
 * \endverbatim
 *
 * @param x a decision vector.
 *
 * @return \p true if the decision vector results in a feasible fitness, \p false otherwise.
 *
 * @throws unspecified any exception thrown by problem::feasibility_f() or problem::fitness().
 */
bool problem::feasibility_x(const vector_double &x) const
{
    // Wrong dimensions of x will trigger exceptions in the called functions
    return feasibility_f(fitness(x));
}

/// Feasibility of a fitness vector.
/**
 * This method will check the feasibility of a fitness vector \p f against
 * the tolerances returned by problem::get_c_tol().
 *
 * @param f a fitness vector.
 *
 * @return \p true if the fitness vector is feasible, \p false otherwise.
 *
 * @throws std::invalid_argument if the size of \p f is not the same as the output of problem::get_nf().
 */
bool problem::feasibility_f(const vector_double &f) const
{
    if (f.size() != get_nf()) {
        pagmo_throw(std::invalid_argument,
                    "The fitness passed as argument has dimension of: " + std::to_string(f.size())
                        + ", while the problem defines a fitness size of: " + std::to_string(get_nf()));
    }
    auto feas_eq
        = detail::test_eq_constraints(f.data() + get_nobj(), f.data() + get_nobj() + get_nec(), get_c_tol().data());
    auto feas_ineq = detail::test_ineq_constraints(f.data() + get_nobj() + get_nec(), f.data() + f.size(),
                                                   get_c_tol().data() + get_nec());
    return feas_eq.first + feas_ineq.first == get_nc();
}

/// Problem's extra info.
/**
 * If the UDP satisfies pagmo::has_extra_info, then this method will return the output of its
 * <tt>%get_extra_info()</tt> method. Otherwise, an empty string will be returned.
 *
 * @return extra info about the UDP.
 *
 * @throws unspecified any exception thrown by the <tt>%get_extra_info()</tt> method of the UDP.
 */
std::string problem::get_extra_info() const
{
    return ptr()->get_extra_info();
}

/// Check if the problem is in a valid state.
/**
 * @return ``false`` if ``this`` was moved from, ``true`` otherwise.
 */
bool problem::is_valid() const
{
    return static_cast<bool>(m_ptr);
}

/// Streaming operator
/**
 * This function will stream to \p os a human-readable representation of the input
 * problem \p p.
 *
 * @param os input <tt>std::ostream</tt>.
 * @param p pagmo::problem object to be streamed.
 *
 * @return a reference to \p os.
 *
 * @throws unspecified any exception thrown by querying various problem properties and streaming them into \p os.
 */
std::ostream &operator<<(std::ostream &os, const problem &p)
{
    os << "Problem name: " << p.get_name();
    if (p.is_stochastic()) {
        stream(os, " [stochastic]");
    }
    os << "\n\tGlobal dimension:\t\t\t" << p.get_nx() << '\n';
    os << "\tInteger dimension:\t\t\t" << p.get_nix() << '\n';
    os << "\tFitness dimension:\t\t\t" << p.get_nf() << '\n';
    os << "\tNumber of objectives:\t\t\t" << p.get_nobj() << '\n';
    os << "\tEquality constraints dimension:\t\t" << p.get_nec() << '\n';
    os << "\tInequality constraints dimension:\t" << p.get_nic() << '\n';
    if (p.get_nec() + p.get_nic() > 0u) {
        stream(os, "\tTolerances on constraints: ", p.get_c_tol(), '\n');
    }
    os << "\tLower bounds: ";
    stream(os, p.get_bounds().first, '\n');
    os << "\tUpper bounds: ";
    stream(os, p.get_bounds().second, '\n');
    stream(os, "\tHas batch fitness evaluation: ", p.has_batch_fitness(), '\n');
    stream(os, "\n\tHas gradient: ", p.has_gradient(), '\n');
    stream(os, "\tUser implemented gradient sparsity: ", p.has_gradient_sparsity(), '\n');
    if (p.has_gradient()) {
        stream(os, "\tExpected gradients: ", p.m_gs_dim, '\n');
    }
    stream(os, "\tHas hessians: ", p.has_hessians(), '\n');
    stream(os, "\tUser implemented hessians sparsity: ", p.has_hessians_sparsity(), '\n');
    if (p.has_hessians()) {
        stream(os, "\tExpected hessian components: ", p.m_hs_dim, '\n');
    }
    stream(os, "\n\tFitness evaluations: ", p.get_fevals(), '\n');
    if (p.has_gradient()) {
        stream(os, "\tGradient evaluations: ", p.get_gevals(), '\n');
    }
    if (p.has_hessians()) {
        stream(os, "\tHessians evaluations: ", p.get_hevals(), '\n');
    }
    stream(os, "\n\tThread safety: ", p.get_thread_safety(), '\n');

    const auto extra_str = p.get_extra_info();
    if (!extra_str.empty()) {
        stream(os, "\nExtra info:\n", extra_str);
    }
    return os;
}

void problem::check_gradient_sparsity(const sparsity_pattern &gs) const
{
    // Cache a couple of quantities.
    const auto nx = get_nx();
    const auto nf = get_nf();

    // Check the pattern.
    for (auto it = gs.begin(); it != gs.end(); ++it) {
        if ((it->first >= nf) || (it->second >= nx)) {
            pagmo_throw(std::invalid_argument, "Invalid pair detected in the gradient sparsity pattern: ("
                                                   + std::to_string(it->first) + ", " + std::to_string(it->second)
                                                   + ")\nFitness dimension is: " + std::to_string(nf)
                                                   + "\nDecision vector dimension is: " + std::to_string(nx));
        }
        if (it == gs.begin()) {
            continue;
        }
        if (!(*(it - 1) < *it)) {
            pagmo_throw(std::invalid_argument,
                        "The gradient sparsity pattern is not strictly sorted in ascending order: the indices pair ("
                            + std::to_string((it - 1)->first) + ", " + std::to_string((it - 1)->second)
                            + ") is greater than or equal to the successive indices pair (" + std::to_string(it->first)
                            + ", " + std::to_string(it->second) + ")");
        }
    }
}

void problem::check_hessians_sparsity(const std::vector<sparsity_pattern> &hs) const
{
    // 1 - We check that a hessian sparsity is provided for each component
    // of the fitness
    const auto nf = get_nf();
    if (hs.size() != nf) {
        pagmo_throw(std::invalid_argument, "Invalid dimension of the hessians_sparsity: " + std::to_string(hs.size())
                                               + ", expected: " + std::to_string(nf));
    }
    // 2 - We check that all hessian sparsity patterns have
    // valid indices.
    for (const auto &one_hs : hs) {
        check_hessian_sparsity(one_hs);
    }
}

void problem::check_hessian_sparsity(const sparsity_pattern &hs) const
{
    const auto nx = get_nx();
    // We check that the hessian sparsity pattern has
    // valid indices. Assuming a lower triangular representation of
    // a symmetric matrix. Example, for a 4x4 dense symmetric
    // [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)]
    for (auto it = hs.begin(); it != hs.end(); ++it) {
        if ((it->first >= nx) || (it->second > it->first)) {
            pagmo_throw(std::invalid_argument, "Invalid pair detected in the hessians sparsity pattern: ("
                                                   + std::to_string(it->first) + ", " + std::to_string(it->second)
                                                   + ")\nDecision vector dimension is: " + std::to_string(nx)
                                                   + "\nNOTE: hessian is a symmetric matrix and PaGMO represents "
                                                     "it as lower triangular: i.e (i,j) is not valid if j>i");
        }
        if (it == hs.begin()) {
            continue;
        }
        if (!(*(it - 1) < *it)) {
            pagmo_throw(std::invalid_argument,
                        "The hessian sparsity pattern is not strictly sorted in ascending order: the indices pair ("
                            + std::to_string((it - 1)->first) + ", " + std::to_string((it - 1)->second)
                            + ") is greater than or equal to the successive indices pair (" + std::to_string(it->first)
                            + ", " + std::to_string(it->second) + ")");
        }
    }
}

void problem::check_gradient_vector(const vector_double &gr) const
{
    // Checks that the gradient vector returned has the same dimensions of the sparsity_pattern
    if (gr.size() != m_gs_dim) {
        pagmo_throw(std::invalid_argument,
                    "Gradients returned: " + std::to_string(gr.size()) + ", should be " + std::to_string(m_gs_dim));
    }
}

void problem::check_hessians_vector(const std::vector<vector_double> &hs) const
{
    // 1 - Check that hs has size get_nf()
    if (hs.size() != get_nf()) {
        pagmo_throw(std::invalid_argument, "The hessians vector has a size of " + std::to_string(hs.size())
                                               + ", but the fitness dimension of the problem is "
                                               + std::to_string(get_nf()) + ". The two values must be equal");
    }
    // 2 - Check that the hessians returned have the same dimensions of the
    // corresponding sparsity patterns
    // NOTE: the dimension of m_hs_dim is guaranteed to be get_nf() on construction.
    for (decltype(hs.size()) i = 0u; i < hs.size(); ++i) {
        if (hs[i].size() != m_hs_dim[i]) {
            pagmo_throw(std::invalid_argument, "On the hessian no. " + std::to_string(i)
                                                   + ": Components returned: " + std::to_string(hs[i].size())
                                                   + ", should be " + std::to_string(m_hs_dim[i]));
        }
    }
}

namespace detail
{

// Check that the decision vector starting at dv and with
// size s is compatible with the input problem p.
void prob_check_dv(const problem &p, const double *dv, vector_double::size_type s)
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
void prob_check_fv(const problem &p, const double *fv, vector_double::size_type s)
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

// Small helper for the invocation of the UDP's batch_fitness() *without* checks.
// This is useful for avoiding doing double checks on the input/output values
// of batch_fitness() when we are sure that the checks have been performed elsewhere already.
// This helper will also take care of increasing the fevals counter in the
// input problem.
vector_double prob_invoke_mem_batch_fitness(const problem &p, const vector_double &dvs)
{
    // Invoke the batch fitness from the UDP.
    auto retval(p.ptr()->batch_fitness(dvs));

    // Increment the number of fitness evaluations.
    p.increment_fevals(boost::numeric_cast<unsigned long long>(dvs.size() / p.get_nx()));

    return retval;
}

} // namespace detail

} // namespace pagmo
