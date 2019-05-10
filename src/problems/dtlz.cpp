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

#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/detail/constants.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

dtlz::dtlz(unsigned prob_id, vector_double::size_type dim, vector_double::size_type fdim, unsigned alpha)
    : m_prob_id(prob_id), m_alpha(alpha), m_dim(dim), m_fdim(fdim)
{
    if (prob_id == 0u || prob_id > 7u) {
        pagmo_throw(std::invalid_argument, "DTLZ test suite contains seven (prob_id = [1 ... 7]) problems, prob_id="
                                               + std::to_string(prob_id) + " was detected");
    }
    if (fdim < 2u) {
        pagmo_throw(std::invalid_argument,
                    "DTLZ test problem have a minimum of 2 objectives: fdim=" + std::to_string(fdim) + " was detected");
    }
    // We conservatively limit these dimensions to avoid checking overflows later
    if (fdim > std::numeric_limits<decltype(fdim)>::max() / 3u) {
        pagmo_throw(std::invalid_argument, "The number of objectives is too large");
    }
    if (dim > std::numeric_limits<decltype(dim)>::max() / 3u) {
        pagmo_throw(std::invalid_argument, "The problem dimension is too large");
    }
    if (dim <= fdim) {
        pagmo_throw(std::invalid_argument, "The problem dimension has to be larger than the number of objectives.");
    }
}

/// Fitness computation
/**
 * Computes the fitness for this UDP
 *
 * @param x the decision vector.
 *
 * @return the fitness of \p x.
 */
vector_double dtlz::fitness(const vector_double &x) const
{
    vector_double retval;
    switch (m_prob_id) {
        case 1:
            retval = f1_objfun_impl(x);
            break;
        case 2:
        case 3:
            retval = f23_objfun_impl(x);
            break;
        case 4:
            retval = f4_objfun_impl(x);
            break;
        case 5:
        case 6:
            retval = f56_objfun_impl(x);
            break;
        case 7:
            retval = f7_objfun_impl(x);
            break;
    }
    return retval;
}

/// Box-bounds
/**
 *
 * It returns the box-bounds for this UDP, [0,1] for each component
 *
 * @return the lower and upper bounds for each of the decision vector components
 */
std::pair<vector_double, vector_double> dtlz::get_bounds() const
{
    return {vector_double(m_dim, 0.), vector_double(m_dim, 1.)};
}

/// Distance from the Pareto front (of a population)
/**
 * Convergence metric for a given population (0 = on the optimal front)
 *
 * Takes the average across the input population of the p_distance
 *
 * @param pop population to be assigned a pareto distance
 * @return the p_distance
 *
 */
double dtlz::p_distance(const pagmo::population &pop) const
{
    double c = 0.0;
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        c += p_distance(pop.get_x()[i]);
    }

    return c / static_cast<double>(pop.size());
}

/// Distance from the Pareto front
/**
 * Convergence metric for a given decision_vector (0 = on the optimal front)
 *
 * Introduced by Martens and Izzo, this metric is able
 * to measure "a distance" of any point from the pareto front of any ZDT
 * problem analytically without the need to precompute the front.
 *
 * @param x input decision vector
 * @return the p_distance
 *
 * See: Märtens, Marcus, and Dario Izzo. "The asynchronous island model
 * and NSGA-II: study of a new migration operator and its performance."
 * Proceedings of the 15th annual conference on Genetic and evolutionary computation. ACM, 2013.
 */
double dtlz::p_distance(const vector_double &x) const
{
    if (x.size() != m_dim) {
        pagmo_throw(std::invalid_argument, "The size of the decision vector should be " + std::to_string(m_dim)
                                               + " while " + std::to_string(x.size()) + " was detected");
    }
    return convergence_metric(x);
}

/// Problem name
/**
 * @return a string containing the problem name
 */
std::string dtlz::get_name() const
{
    return "DTLZ" + std::to_string(m_prob_id);
}

/// Object serialization
/**
 * This method will save/load \p this into the archive \p ar.
 *
 * @param ar target archive.
 *
 * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
 */
template <typename Archive>
void dtlz::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_prob_id, m_dim, m_fdim, m_alpha);
}

// Convergence metric for a dv (0 = converged to the optimal front)
double dtlz::g_func(const vector_double &x) const
{
    double retval = 0.;
    switch (m_prob_id) { // We start with the 6-7 cases as for absurd reasons behind my comprehension this is
                         // way more efficient
        case 6:
            retval = g6_func(x);
            break;
        case 7:
            retval = g7_func(x);
            break;
        case 1:
        case 3:
            retval = g13_func(x);
            break;
        case 2:
        case 4:
        case 5:
            retval = g245_func(x);
            break;
    }
    return retval;
}

// Implementations of the different g-functions used
double dtlz::g13_func(const vector_double &x) const
{
    double y = 0.;
    for (decltype(x.size()) i = 0u; i < x.size(); ++i) {
        y += std::pow(x[i] - 0.5, 2) - std::cos(20. * detail::pi() * (x[i] - 0.5));
    }
    return 100. * (y + static_cast<double>(x.size()));
}

double dtlz::g245_func(const vector_double &x) const
{
    double y = 0.;
    for (decltype(x.size()) i = 0u; i < x.size(); ++i) {
        y += std::pow(x[i] - 0.5, 2);
    }
    return y;
}

double dtlz::g6_func(const vector_double &x) const
{
    double y = 0.0;
    for (decltype(x.size()) i = 0u; i < x.size(); ++i) {
        y += std::pow(x[i], 0.1);
    }
    return y;
}

double dtlz::g7_func(const vector_double &x) const
{
    // NOTE: the original g-function should return 1 + (9.0 / x.size()) * y but we drop the 1
    // to have the minimum at 0.0 so we can use the p_distance implementation in base_dtlz
    // to have the p_distance converging towards 0.0 rather then towards 1.0
    double y = 0.;
    for (decltype(x.size()) i = 0u; i < x.size(); ++i) {
        y += x[i];
    }
    return (9. / static_cast<double>(x.size())) * y;
}

// Implementation of the distribution function h
double dtlz::h7_func(const vector_double &f, double g) const
{
    // NOTE: we intentionally ignore the last element of the vector to make things easier
    double y = 0.;

    for (decltype(f.size()) i = 0u; i < f.size() - 1; ++i) {
        y += (f[i] / (1.0 + g)) * (1.0 + std::sin(3 * detail::pi() * f[i]));
    }
    return static_cast<double>(m_fdim) - y;
}

vector_double dtlz::f1_objfun_impl(const vector_double &x) const
{
    vector_double f(m_fdim);
    // computing distance-function
    vector_double x_M;

    for (decltype(x.size()) i = f.size() - 1u; i < x.size(); ++i) {
        x_M.push_back(x[i]);
    }

    double g = g_func(x_M);

    // computing shape-functions
    f[0] = 0.5 * (1. + g);

    for (decltype(f.size()) i = 0u; i < f.size() - 1u; ++i) {
        f[0] *= x[i];
    }

    for (decltype(f.size()) i = 1u; i < f.size() - 1u; ++i) {
        f[i] = 0.5 * (1.0 + g);
        for (decltype(f.size()) j = 0u; j < f.size() - (i + 1); ++j) {
            f[i] *= x[j];
        }
        f[i] *= 1. - x[f.size() - (i + 1u)];
    }

    f[f.size() - 1u] = 0.5 * (1. - x[0]) * (1. + g);
    return f;
}

vector_double dtlz::f23_objfun_impl(const vector_double &x) const
{
    vector_double f(m_fdim);
    // computing distance-function
    vector_double x_M;
    for (decltype(x.size()) i = f.size() - 1u; i < x.size(); ++i) {
        x_M.push_back(x[i]);
    }

    auto g = g_func(x_M);

    // computing shape-functions
    f[0] = (1. + g);
    for (decltype(f.size()) i = 0u; i < f.size() - 1u; ++i) {
        f[0] *= cos(x[i] * detail::pi_half());
    }

    for (decltype(f.size()) i = 1u; i < f.size() - 1u; ++i) {
        f[i] = (1. + g);
        for (decltype(f.size()) j = 0u; j < f.size() - (i + 1u); ++j) {
            f[i] *= cos(x[j] * detail::pi_half());
        }
        f[i] *= std::sin(x[f.size() - (i + 1u)] * detail::pi_half());
    }

    f[f.size() - 1u] = (1. + g) * std::sin(x[0] * detail::pi_half());
    return f;
}

vector_double dtlz::f4_objfun_impl(const vector_double &x) const
{
    vector_double f(m_fdim);
    // computing distance-function
    vector_double x_M;
    for (decltype(x.size()) i = f.size() - 1; i < x.size(); ++i) {
        x_M.push_back(x[i]);
    }

    auto g = g_func(x_M);

    // computing shape-functions
    f[0] = (1. + g);
    for (decltype(f.size()) i = 0u; i < f.size() - 1u; ++i) {
        f[0] *= std::cos(std::pow(x[i], m_alpha) * detail::pi_half());
    }

    for (decltype(f.size()) i = 1u; i < f.size() - 1u; ++i) {
        f[i] = (1.0 + g);
        for (decltype(f.size()) j = 0u; j < f.size() - (i + 1u); ++j) {
            f[i] *= std::cos(std::pow(x[j], m_alpha) * detail::pi_half());
        }
        f[i] *= std::sin(std::pow(x[f.size() - (i + 1u)], m_alpha) * detail::pi_half());
    }

    f[f.size() - 1u] = (1. + g) * std::sin(std::pow(x[0], m_alpha) * detail::pi_half());
    return f;
}

vector_double dtlz::f56_objfun_impl(const vector_double &x) const
{
    vector_double f(m_fdim);
    // computing distance-function
    vector_double x_M;

    for (decltype(x.size()) i = f.size() - 1u; i < x.size(); ++i) {
        x_M.push_back(x[i]);
    }

    auto g = g_func(x_M);

    // computing meta-variables
    vector_double theta(f.size(), 0.);
    double t;

    theta[0] = x[0];
    t = 1. / (2. * (1. + g));

    for (decltype(f.size()) i = 1u; i < f.size(); ++i) {
        theta[i] = t + ((g * x[i]) / (1.0 + g));
    }

    // computing shape-functions
    f[0] = (1. + g);
    for (decltype(f.size()) i = 0u; i < f.size() - 1u; ++i) {
        f[0] *= std::cos(theta[i] * detail::pi_half());
    }

    for (decltype(f.size()) i = 1u; i < f.size() - 1u; ++i) {
        f[i] = (1. + g);
        for (decltype(f.size()) j = 0u; j < f.size() - (i + 1u); ++j) {
            f[i] *= std::cos(theta[j] * detail::pi_half());
        }
        f[i] *= std::sin(theta[f.size() - (i + 1u)] * detail::pi_half());
    }

    f[f.size() - 1u] = (1. + g) * std::sin(theta[0] * detail::pi_half());
    return f;
}

vector_double dtlz::f7_objfun_impl(const vector_double &x) const
{
    vector_double f(m_fdim);
    // computing distance-function
    vector_double x_M;
    double g;

    for (decltype(x.size()) i = f.size() - 1u; i < x.size(); ++i) {
        x_M.push_back(x[i]);
    }

    g = 1. + g_func(x_M); // +1.0 according to the original definition of the g-function for DTLZ7

    // computing shape-functions
    for (decltype(f.size()) i = 0u; i < f.size() - 1u; ++i) {
        f[i] = x[i];
    }

    f[f.size() - 1u] = (1. + g) * h7_func(f, g);
    return f;
}

// Gives a convergence metric for the population (0 = converged to the optimal front)
double dtlz::convergence_metric(const vector_double &x) const
{
    double c = 0.;
    vector_double x_M;
    for (decltype(x.size()) j = m_fdim - 1u; j < x.size(); ++j) {
        x_M.push_back(x[j]);
    }
    c += g_func(x_M);
    return c;
}

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::dtlz)
