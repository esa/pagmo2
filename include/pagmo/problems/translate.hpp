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

#ifndef PAGMO_PROBLEM_TRANSLATE_HPP
#define PAGMO_PROBLEM_TRANSLATE_HPP

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <type_traits>

#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
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
class translate
{
    // Enabler for the ctor from UDP or problem. In this case we also allow construction from type problem.
    template <typename T>
    using ctor_enabler = enable_if_t<std::is_constructible<problem, T &&>::value, int>;

public:
    /// Default constructor.
    /**
     * The default constructor will initialize a non-translated pagmo::null_problem.
     */
    translate() : m_problem(null_problem{}), m_translation({0.}) {}

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
        if (translation.size() != m_problem.get_nx()) {
            pagmo_throw(std::invalid_argument,
                        "Length of shift vector is: " + std::to_string(translation.size())
                            + " while the problem dimension is: " + std::to_string(m_problem.get_nx()));
        }
    }

    /// Fitness.
    /**
     * The fitness computation is forwarded to the inner UDP, after the translation of \p x.
     *
     * @param x the decision vector.
     *
     * @return the fitness of \p x.
     *
     * @throws unspecified any exception thrown by memory errors in standard containers,
     * or by problem::fitness().
     */
    vector_double fitness(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return m_problem.fitness(x_deshifted);
    }

    /// Box-bounds.
    /**
     * The box-bounds returned by this method are the translated box-bounds of the inner UDP.
     *
     * @return the lower and upper bounds for each of the decision vector components.
     *
     * @throws unspecified any exception thrown by memory errors in standard containers,
     * or by problem::get_bounds().
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        auto b_sh = m_problem.get_bounds();
        // NOTE: this should be safe as the translation vector has been checked against the
        // bounds size upon construction (via get_nx()).
        return {apply_translation(b_sh.first), apply_translation(b_sh.second)};
    }

    /// Number of objectives.
    /**
     * @return the number of objectives of the inner problem.
     */
    vector_double::size_type get_nobj() const
    {
        return m_problem.get_nobj();
    }

    /// Equality constraint dimension.
    /**
     * Returns the number of equality constraints of the inner problem.
     *
     * @return the number of equality constraints of the inner problem.
     */
    vector_double::size_type get_nec() const
    {
        return m_problem.get_nec();
    }

    /// Inequality constraint dimension.
    /**
     * Returns the number of inequality constraints of the inner problem.
     *
     * @return the number of inequality constraints of the inner problem.
     */
    vector_double::size_type get_nic() const
    {
        return m_problem.get_nic();
    }

    /// Integer dimension
    /**
     * @return the integer dimension of the inner problem.
     */
    vector_double::size_type get_nix() const
    {
        return m_problem.get_nix();
    }

    /// Checks if the inner problem has gradients.
    /**
     * The <tt>has_gradient()</tt> computation is forwarded to the inner problem.
     *
     * @return a flag signalling the availability of the gradient in the inner problem.
     */
    bool has_gradient() const
    {
        return m_problem.has_gradient();
    }

    /// Gradients.
    /**
     * The gradients computation is forwarded to the inner problem, after the translation of \p x.
     *
     * @param x the decision vector.
     *
     * @return the gradient of the fitness function.
     *
     * @throws unspecified any exception thrown by memory errors in standard containers,
     * or by <tt>problem::gradient()</tt>.
     */
    vector_double gradient(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return m_problem.gradient(x_deshifted);
    }

    /// Checks if the inner problem has gradient sparisty implemented.
    /**
     * The <tt>has_gradient_sparsity()</tt> computation is forwarded to the inner problem.
     *
     * @return a flag signalling the availability of the gradient sparisty in the inner problem.
     */
    bool has_gradient_sparsity() const
    {
        return m_problem.has_gradient_sparsity();
    }

    /// Gradient sparsity.
    /**
     * The <tt>gradient_sparsity</tt> computation is forwarded to the inner problem.
     *
     * @return the gradient sparsity of the inner problem.
     */
    sparsity_pattern gradient_sparsity() const
    {
        return m_problem.gradient_sparsity();
    }

    /// Checks if the inner problem has hessians.
    /**
     * The <tt>has_hessians()</tt> computation is forwarded to the inner problem.
     *
     * @return a flag signalling the availability of the hessians in the inner problem.
     */
    bool has_hessians() const
    {
        return m_problem.has_hessians();
    }

    /// Hessians.
    /**
     * The <tt>hessians()</tt> computation is forwarded to the inner problem, after the translation of \p x.
     *
     * @param x the decision vector.
     *
     * @return the hessians of the fitness function computed at \p x.
     *
     * @throws unspecified any exception thrown by memory errors in standard containers,
     * or by problem::hessians().
     */
    std::vector<vector_double> hessians(const vector_double &x) const
    {
        vector_double x_deshifted = translate_back(x);
        return m_problem.hessians(x_deshifted);
    }

    /// Checks if the inner problem has hessians sparisty implemented.
    /**
     * The <tt>has_hessians_sparsity()</tt> computation is forwarded to the inner problem.
     *
     * @return a flag signalling the availability of the hessians sparisty in the inner problem.
     */
    bool has_hessians_sparsity() const
    {
        return m_problem.has_hessians_sparsity();
    }

    /// Hessians sparsity.
    /**
     * The <tt>hessians_sparsity()</tt> computation is forwarded to the inner problem.
     *
     * @return the hessians sparsity of the inner problem.
     */
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return m_problem.hessians_sparsity();
    }

    /// Calls <tt>has_set_seed()</tt> of the inner problem.
    /**
     * Calls the method <tt>has_set_seed()</tt> of the inner problem.
     *
     * @return a flag signalling wether the inner problem is stochastic.
     */
    bool has_set_seed() const
    {
        return m_problem.has_set_seed();
    }

    /// Calls <tt>set_seed()</tt> of the inner problem.
    /**
     * Calls the method <tt>set_seed()</tt> of the inner problem.
     *
     * @param seed seed to be set.
     *
     * @throws unspecified any exception thrown by the method <tt>set_seed()</tt> of the inner problem.
     */
    void set_seed(unsigned seed)
    {
        return m_problem.set_seed(seed);
    }

    /// Problem name
    /**
     * This method will add <tt>[translated]</tt> to the name provided by the inner problem.
     *
     * @return a string containing the problem name.
     *
     * @throws unspecified any exception thrown by <tt>problem::get_name()</tt> or memory errors in standard classes.
     */
    std::string get_name() const
    {
        return m_problem.get_name() + " [translated]";
    }

    /// Extra info
    /**
     * This method will append a description of the translation vector to the extra info provided
     * by the inner problem.
     *
     * @return a string containing extra info on the problem.
     *
     * @throws unspecified any exception thrown by problem::get_extra_info(), the public interface of
     * \p std::ostringstream or memory errors in standard classes.
     */
    std::string get_extra_info() const
    {
        std::ostringstream oss;
        stream(oss, m_translation);
        return m_problem.get_extra_info() + "\n\tTranslation Vector: " + oss.str();
    }

    /// Get the translation vector
    /**
     * @return a reference to the translation vector.
     */
    const vector_double &get_translation() const
    {
        return m_translation;
    }

    /// Problem's thread safety level.
    /**
     * The thread safety of a meta-problem is defined by the thread safety of the inner pagmo::problem.
     *
     * @return the thread safety level of the inner pagmo::problem.
     */
    thread_safety get_thread_safety() const
    {
        return m_problem.get_thread_safety();
    }

    /// Getter for the inner problem.
    /**
     * Returns a const reference to the inner pagmo::problem.
     *
     * @return a const reference to the inner pagmo::problem.
     */
    const problem &get_inner_problem() const
    {
        return m_problem;
    }

    /// Getter for the inner problem.
    /**
     * Returns a reference to the inner pagmo::problem.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    The ability to extract a non const reference is provided only in order to allow to call
     *    non-const methods on the internal :cpp:class:`pagmo::problem` instance. Assigning a new
     *    :cpp:class:`pagmo::problem` via this reference is undefined behaviour.
     *
     * \endverbatim
     *
     * @return a reference to the inner pagmo::problem.
     */
    problem &get_inner_problem()
    {
        return m_problem;
    }

    /// Object serialization
    /**
     * This method will save/load \p this into/from the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the inner problem and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_problem, m_translation);
    }

private:
    vector_double translate_back(const vector_double &x) const
    {
        // NOTE: here we use assert instead of throwing because the general idea is that we don't
        // protect UDPs from misuses, and we have checks in problem. In Python, UDP methods that could cause
        // troubles are not exposed.
        assert(x.size() == m_translation.size());
        vector_double x_sh(x.size());
        std::transform(x.begin(), x.end(), m_translation.begin(), x_sh.begin(), std::minus<double>());
        return x_sh;
    }

    vector_double apply_translation(const vector_double &x) const
    {
        assert(x.size() == m_translation.size());
        vector_double x_sh(x.size());
        std::transform(x.begin(), x.end(), m_translation.begin(), x_sh.begin(), std::plus<double>());
        return x_sh;
    }
    // Inner problem
    problem m_problem;
    // translation vector
    vector_double m_translation;
};
} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::translate)

#endif
