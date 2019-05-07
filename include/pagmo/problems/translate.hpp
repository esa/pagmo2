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

#ifndef PAGMO_PROBLEMS_TRANSLATE_HPP
#define PAGMO_PROBLEMS_TRANSLATE_HPP

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// The translate meta-problem.
/**
 * This meta-problem translates the whole search space of an input problem
 * by a fixed translation vector. pagmo::translate objects are user-defined problems that can be used in
 * the definition of a pagmo::problem.
 */
class PAGMO_DLL_PUBLIC translate
{
public:
    // Default constructor.
    translate();

private:
    // Enabler for the ctor from UDP or problem. In this case we also allow construction from type problem.
    template <typename T>
    using ctor_enabler = enable_if_t<std::is_constructible<problem, T &&>::value, int>;
    // Implementation of the generic ctor.
    void generic_ctor_impl(const vector_double &);

public:
    /// Constructor from problem and translation vector.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if ``T`` can be used to construct a :cpp:class:`pagmo::problem`.
     *
     * \endverbatim
     *
     * Wraps a user-defined problem so that its fitness , bounds, etc. will be shifted by a
     * translation vector.
     *
     * @param p a pagmo::problem or a user-defined problem (UDP).
     * @param translation an <tt>std::vector</tt> containing the translation to apply.
     *
     * @throws std::invalid_argument if the length of \p translation is
     * not equal to the problem dimension \f$ n_x\f$.
     * @throws unspecified any exception thrown by the pagmo::problem constructor.
     */
    template <typename T, ctor_enabler<T> = 0>
    explicit translate(T &&p, const vector_double &translation)
        : m_problem(std::forward<T>(p)), m_translation(translation)
    {
        generic_ctor_impl(translation);
    }

    // Fitness.
    vector_double fitness(const vector_double &) const;

    // Batch fitness.
    vector_double batch_fitness(const vector_double &) const;

    // Check if the inner problem can compute fitnesses in batch mode.
    bool has_batch_fitness() const;

    // Box-bounds.
    std::pair<vector_double, vector_double> get_bounds() const;

    // Number of objectives.
    vector_double::size_type get_nobj() const;

    // Equality constraint dimension.
    vector_double::size_type get_nec() const;

    // Inequality constraint dimension.
    vector_double::size_type get_nic() const;

    // Integer dimension
    vector_double::size_type get_nix() const;

    // Checks if the inner problem has gradients.
    bool has_gradient() const;

    // Gradients.
    vector_double gradient(const vector_double &) const;

    // Checks if the inner problem has gradient sparisty implemented.
    bool has_gradient_sparsity() const;

    // Gradient sparsity.
    sparsity_pattern gradient_sparsity() const;

    // Checks if the inner problem has hessians.
    bool has_hessians() const;

    // Hessians.
    std::vector<vector_double> hessians(const vector_double &) const;

    // Checks if the inner problem has hessians sparisty implemented.
    bool has_hessians_sparsity() const;

    // Hessians sparsity.
    std::vector<sparsity_pattern> hessians_sparsity() const;

    // Calls <tt>has_set_seed()</tt> of the inner problem.
    bool has_set_seed() const;

    // Calls <tt>set_seed()</tt> of the inner problem.
    void set_seed(unsigned);

    // Problem name
    std::string get_name() const;

    // Extra info
    std::string get_extra_info() const;

    // Get the translation vector
    const vector_double &get_translation() const;

    // Problem's thread safety level.
    thread_safety get_thread_safety() const;

    // Getter for the inner problem.
    const problem &get_inner_problem() const;

    // Getter for the inner problem.
    problem &get_inner_problem();

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    PAGMO_DLL_LOCAL vector_double translate_back(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double apply_translation(const vector_double &) const;

    // Inner problem
    problem m_problem;
    // translation vector
    vector_double m_translation;
};

} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::translate)

#endif
