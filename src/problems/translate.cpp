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
#include <functional>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#pragma GCC diagnostic ignored "-Wsuggest-attribute=const"
#endif

namespace pagmo
{

/// Default constructor.
/**
 * The constructor will initialize a non-translated default-constructed pagmo::problem.
 */
translate::translate() : m_translation({0.}) {}

void translate::generic_ctor_impl(const vector_double &translation)
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
vector_double translate::fitness(const vector_double &x) const
{
    vector_double x_deshifted = translate_back(x);
    return m_problem.fitness(x_deshifted);
}

/// Batch fitness.
/**
 * The batch fitness computation is forwarded to the inner UDP, after the translation of \p xs.
 *
 * @param xs the input decision vectors.
 *
 * @return the fitnesses of \p xs.
 *
 * @throws unspecified any exception thrown by memory errors in standard containers,
 * threading primitives, or by problem::batch_fitness().
 */
vector_double translate::batch_fitness(const vector_double &xs) const
{
    const auto nx = m_problem.get_nx();
    // Assume xs is sane.
    assert(xs.size() % nx == 0u);
    const auto n_dvs = xs.size() / nx;

    // Prepare the deshifted dvs.
    vector_double xs_deshifted(xs.size());

    // Do the deshifting in parallel.
    using range_t = tbb::blocked_range<decltype(xs.size())>;
    tbb::parallel_for(range_t(0, n_dvs), [&xs, &xs_deshifted, nx, this](const range_t &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
#if defined(_MSC_VER)
            std::transform(stdext::make_unchecked_array_iterator(xs.data() + i * nx),
                           stdext::make_unchecked_array_iterator(xs.data() + (i + 1u) * nx),
                           stdext::make_unchecked_array_iterator(m_translation.data()),
                           stdext::make_unchecked_array_iterator(xs_deshifted.data() + i * nx), std::minus<double>{});
#else
                std::transform(xs.data() + i * nx, xs.data() + (i + 1u) * nx, m_translation.data(),
                               xs_deshifted.data() + i * nx, std::minus<double>{});
#endif
        }
    });

    // Invoke batch_fitness() from m_problem.
    // NOTE: in non-debug mode, use the helper that avoids calling the checks in m_problem.batch_fitness().
    // The translate metaproblem does not change the dimensionality of the problem
    // or of the fitness, thus all the checks run by m_problem.batch_fitness()
    // are redundant.
#if defined(NDEBUG)
    return detail::prob_invoke_mem_batch_fitness(m_problem, xs_deshifted);
#else
    return m_problem.batch_fitness(xs_deshifted);
#endif
}

/// Check if the inner problem can compute fitnesses in batch mode.
/**
 * @return the output of the <tt>has_batch_fitness()</tt> member function invoked
 * by the inner problem.
 */
bool translate::has_batch_fitness() const
{
    return m_problem.has_batch_fitness();
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
std::pair<vector_double, vector_double> translate::get_bounds() const
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
vector_double::size_type translate::get_nobj() const
{
    return m_problem.get_nobj();
}

/// Equality constraint dimension.
/**
 * Returns the number of equality constraints of the inner problem.
 *
 * @return the number of equality constraints of the inner problem.
 */
vector_double::size_type translate::get_nec() const
{
    return m_problem.get_nec();
}

/// Inequality constraint dimension.
/**
 * Returns the number of inequality constraints of the inner problem.
 *
 * @return the number of inequality constraints of the inner problem.
 */
vector_double::size_type translate::get_nic() const
{
    return m_problem.get_nic();
}

/// Integer dimension
/**
 * @return the integer dimension of the inner problem.
 */
vector_double::size_type translate::get_nix() const
{
    return m_problem.get_nix();
}

/// Checks if the inner problem has gradients.
/**
 * The <tt>has_gradient()</tt> computation is forwarded to the inner problem.
 *
 * @return a flag signalling the availability of the gradient in the inner problem.
 */
bool translate::has_gradient() const
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
vector_double translate::gradient(const vector_double &x) const
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
bool translate::has_gradient_sparsity() const
{
    return m_problem.has_gradient_sparsity();
}

/// Gradient sparsity.
/**
 * The <tt>gradient_sparsity</tt> computation is forwarded to the inner problem.
 *
 * @return the gradient sparsity of the inner problem.
 */
sparsity_pattern translate::gradient_sparsity() const
{
    return m_problem.gradient_sparsity();
}

/// Checks if the inner problem has hessians.
/**
 * The <tt>has_hessians()</tt> computation is forwarded to the inner problem.
 *
 * @return a flag signalling the availability of the hessians in the inner problem.
 */
bool translate::has_hessians() const
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
std::vector<vector_double> translate::hessians(const vector_double &x) const
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
bool translate::has_hessians_sparsity() const
{
    return m_problem.has_hessians_sparsity();
}

/// Hessians sparsity.
/**
 * The <tt>hessians_sparsity()</tt> computation is forwarded to the inner problem.
 *
 * @return the hessians sparsity of the inner problem.
 */
std::vector<sparsity_pattern> translate::hessians_sparsity() const
{
    return m_problem.hessians_sparsity();
}

/// Calls <tt>has_set_seed()</tt> of the inner problem.
/**
 * Calls the method <tt>has_set_seed()</tt> of the inner problem.
 *
 * @return a flag signalling wether the inner problem is stochastic.
 */
bool translate::has_set_seed() const
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
void translate::set_seed(unsigned seed)
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
std::string translate::get_name() const
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
std::string translate::get_extra_info() const
{
    std::ostringstream oss;
    stream(oss, m_translation);
    return m_problem.get_extra_info() + "\n\tTranslation Vector: " + oss.str();
}

/// Get the translation vector
/**
 * @return a reference to the translation vector.
 */
const vector_double &translate::get_translation() const
{
    return m_translation;
}

/// Problem's thread safety level.
/**
 * The thread safety of a meta-problem is defined by the thread safety of the inner pagmo::problem.
 *
 * @return the thread safety level of the inner pagmo::problem.
 */
thread_safety translate::get_thread_safety() const
{
    return m_problem.get_thread_safety();
}

/// Getter for the inner problem.
/**
 * Returns a const reference to the inner pagmo::problem.
 *
 * @return a const reference to the inner pagmo::problem.
 */
const problem &translate::get_inner_problem() const
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
problem &translate::get_inner_problem()
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
void translate::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_problem, m_translation);
}

vector_double translate::translate_back(const vector_double &x) const
{
    // NOTE: here we use assert instead of throwing because the general idea is that we don't
    // protect UDPs from misuses, and we have checks in problem. In Python, UDP methods that could cause
    // troubles are not exposed.
    assert(x.size() == m_translation.size());
    vector_double x_sh(x.size());
    std::transform(x.begin(), x.end(), m_translation.begin(), x_sh.begin(), std::minus<double>());
    return x_sh;
}

vector_double translate::apply_translation(const vector_double &x) const
{
    assert(x.size() == m_translation.size());
    vector_double x_sh(x.size());
    std::transform(x.begin(), x.end(), m_translation.begin(), x_sh.begin(), std::plus<double>());
    return x_sh;
}

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::translate)
