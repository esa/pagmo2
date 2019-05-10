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
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/constants.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

zdt::zdt(unsigned prob_id, unsigned param) : m_prob_id(prob_id), m_param(param)
{
    if (param < 2u) {
        pagmo_throw(std::invalid_argument, "ZDT test problems must have a minimum value of 2 for the constructing "
                                           "parameter (representing the dimension except for ZDT5), "
                                               + std::to_string(param) + " requested");
    }
    if (prob_id == 0u || prob_id > 6u) {
        pagmo_throw(std::invalid_argument, "ZDT test suite contains six (prob_id=[1 ... 6]) problems, prob_id="
                                               + std::to_string(prob_id) + " was detected");
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
vector_double zdt::fitness(const vector_double &x) const
{
    vector_double retval;
    switch (m_prob_id) {
        case 1u:
            retval = zdt1_fitness(x);
            break;
        case 2u:
            retval = zdt2_fitness(x);
            break;
        case 3u:
            retval = zdt3_fitness(x);
            break;
        case 4u:
            retval = zdt4_fitness(x);
            break;
        case 5u:
            retval = zdt5_fitness(x);
            break;
        case 6u:
            retval = zdt6_fitness(x);
            break;
    }
    return retval;
}

/// Box-bounds
/**
 * It returns the box-bounds for this UDP.
 *
 * @return the lower and upper bounds for each of the decision vector components
 */
std::pair<vector_double, vector_double> zdt::get_bounds() const
{
    std::pair<vector_double, vector_double> retval;
    switch (m_prob_id) {
        case 1u:
        case 2u:
        case 3u:
        case 6u:
            retval = {vector_double(m_param, 0.), vector_double(m_param, 1.)};
            break;
        case 4u: {
            vector_double lb(m_param, -5.);
            vector_double ub(m_param, 5.);
            lb[0] = 0.0;
            ub[0] = 1.0;
            retval = {lb, ub};
            break;
        }
        case 5u: {
            auto dim = 30u + 5u * (m_param - 1u);
            retval = {vector_double(dim, 0.), vector_double(dim, 1.)};
            break;
        }
    }
    return retval;
}

/// Integer dimension
/**
 * It returns the integer dimension for this UDP.
 *
 * @return the integer dimension of the UDP
 */
vector_double::size_type zdt::get_nix() const
{
    vector_double::size_type retval = 0u;
    switch (m_prob_id) {
        case 1u:
        case 2u:
        case 3u:
        case 4u:
        case 6u:
            retval = 0u;
            break;
        case 5u: {
            retval = 30u + 5u * (m_param - 1u);
            break;
        }
    }
    return retval;
}

/// Problem name
/**
 * @return a string containing the problem name
 */
std::string zdt::get_name() const
{
    return "ZDT" + std::to_string(m_prob_id);
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
double zdt::p_distance(const population &pop) const
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
double zdt::p_distance(const vector_double &x) const
{
    double retval = 0.;
    switch (m_prob_id) {
        case 1u:
        case 2u:
        case 3u:
            retval = zdt123_p_distance(x);
            break;
        case 4u:
            retval = zdt4_p_distance(x);
            break;
        case 5u:
            retval = zdt5_p_distance(x);
            break;
        case 6u:
            retval = zdt6_p_distance(x);
            break;
    }
    return retval;
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
void zdt::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_prob_id, m_param);
}

vector_double zdt::zdt1_fitness(const vector_double &x) const
{
    double g = 0.;
    vector_double f(2, 0.);
    f[0] = x[0];
    auto N = x.size();

    for (decltype(N) i = 1u; i < N; ++i) {
        g += x[i];
    }
    g = 1. + (9. * g) / static_cast<double>(N - 1u);

    f[1] = g * (1. - std::sqrt(x[0] / g));
    return f;
}

vector_double zdt::zdt2_fitness(const vector_double &x) const
{
    double g = 0.;
    vector_double f(2, 0.);
    f[0] = x[0];
    auto N = x.size();

    for (decltype(N) i = 1u; i < N; ++i) {
        g += x[i];
    }
    g = 1. + (9. * g) / static_cast<double>(N - 1u);
    f[1] = g * (1. - (x[0] / g) * (x[0] / g));

    return f;
}

vector_double zdt::zdt3_fitness(const vector_double &x) const
{
    double g = 0.;
    vector_double f(2, 0.);
    f[0] = x[0];
    auto N = x.size();

    for (decltype(N) i = 1u; i < N; ++i) {
        g += x[i];
    }
    g = 1. + (9. * g) / static_cast<double>(N - 1u);
    f[1] = g * (1. - std::sqrt(x[0] / g) - x[0] / g * std::sin(10. * detail::pi() * x[0]));

    return f;
}

vector_double zdt::zdt4_fitness(const vector_double &x) const
{
    double g = 0.;
    vector_double f(2, 0.);
    f[0] = x[0];
    auto N = x.size();

    g = 1 + 10 * static_cast<double>(N - 1u);
    f[0] = x[0];
    for (decltype(N) i = 1u; i < N; ++i) {
        g += x[i] * x[i] - 10. * std::cos(4. * detail::pi() * x[i]);
    }
    f[1] = g * (1. - std::sqrt(x[0] / g));

    return f;
}

vector_double zdt::zdt5_fitness(const vector_double &x_double) const
{
    double g = 0.;
    vector_double f(2, 0.);
    auto size_x = x_double.size();
    auto n_vectors = ((size_x - 30u) / 5u) + 1u;

    unsigned k = 30;
    std::vector<vector_double::size_type> u(n_vectors, 0u);
    std::vector<vector_double::size_type> v(n_vectors);

    // Convert the input vector into rounded values (integers)
    vector_double x;
    std::transform(x_double.begin(), x_double.end(), std::back_inserter(x),
                   [](double item) { return std::round(item); });
    f[0] = x[0];

    // Counts how many 1s are there in the first (30 dim)
    u[0] = static_cast<vector_double::size_type>(std::count(x.begin(), x.begin() + 30, 1.));

    for (decltype(n_vectors) i = 1u; i < n_vectors; ++i) {
        for (int j = 0; j < 5; ++j) {
            if (x[k] == 1.) {
                ++u[i];
            }
            ++k;
        }
    }
    f[0] = 1.0 + static_cast<double>(u[0]);
    for (decltype(n_vectors) i = 1u; i < n_vectors; ++i) {
        if (u[i] < 5u) {
            v[i] = 2u + u[i];
        } else {
            v[i] = 1u;
        }
    }
    for (decltype(n_vectors) i = 1u; i < n_vectors; ++i) {
        g += static_cast<double>(v[i]);
    }
    f[1] = g * (1. / f[0]);
    return f;
}

vector_double zdt::zdt6_fitness(const vector_double &x) const
{
    double g = 0.;
    vector_double f(2, 0.);
    f[0] = x[0];
    auto N = x.size();

    f[0] = 1 - std::exp(-4 * x[0]) * std::pow(std::sin(6 * detail::pi() * x[0]), 6);
    for (decltype(N) i = 1; i < N; ++i) {
        g += x[i];
    }
    g = 1 + 9 * std::pow((g / static_cast<double>(N - 1u)), 0.25);
    f[1] = g * (1 - (f[0] / g) * (f[0] / g));

    return f;
}

double zdt::zdt123_p_distance(const vector_double &x) const
{
    double c = 0.;
    double g = 0.;
    auto N = x.size();

    for (decltype(N) j = 1u; j < N; ++j) {
        g += x[j];
    }
    c += 1. + (9. * g) / static_cast<double>(N - 1u);
    return c - 1.;
}

double zdt::zdt4_p_distance(const vector_double &x) const
{
    double c = 0.;
    double g = 0.;
    auto N = x.size();

    for (decltype(N) j = 1u; j < N; ++j) {
        g += x[j] * x[j] - 10. * std::cos(4. * detail::pi() * x[j]);
    }
    c += 1. + 10. * static_cast<double>(N - 1u) + g;
    return c - 1.;
}

double zdt::zdt5_p_distance(const vector_double &x_double) const
{
    // Convert the input vector into floored values (integers)
    vector_double x;
    std::transform(x_double.begin(), x_double.end(), std::back_inserter(x),
                   [](double item) { return std::floor(item); });
    double c = 0.;
    double g = 0.;
    unsigned k = 30;
    auto N = x.size();

    auto n_vectors = (N - 30u) / 5u + 1u;
    std::vector<vector_double::size_type> u(n_vectors, 0);
    std::vector<vector_double::size_type> v(n_vectors);

    for (decltype(n_vectors) i = 1u; i < n_vectors; ++i) {
        for (int j = 0; j < 5; ++j) {
            if (x[k] == 1.) {
                ++u[i];
            }
            ++k;
        }
    }

    for (decltype(n_vectors) i = 1u; i < n_vectors; ++i) {
        if (u[i] < 5u) {
            v[i] = 2u + u[i];
        } else {
            v[i] = 1u;
        }
    }

    for (decltype(n_vectors) i = 1u; i < n_vectors; ++i) {
        g += static_cast<double>(v[i]);
    }
    c += g;
    return c - static_cast<double>(n_vectors) + 1;
}

double zdt::zdt6_p_distance(const vector_double &x) const
{
    double c = 0.;
    double g = 0.;
    auto N = x.size();

    for (decltype(N) j = 1; j < N; ++j) {
        g += x[j];
    }
    c += 1. + 9. * std::pow((g / static_cast<double>(N - 1u)), 0.25);
    return c - 1;
}

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::zdt)
