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
#include <stdexcept>
#include <string>
#include <vector>

#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/detail/prime_numbers.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/utils/discrepancy.hpp>

namespace pagmo
{

/// Sample from a simplex
/**
 * Samples a point on a \f$n\f$ dimensional simplex from a \f$n-1\f$ dimensional point
 *
 * In order to generate a uniform distribution on a simplex, that is to sample a \f$n\f$-dimensional
 * point \f$\mathbf x\f$ such that \f$\sum_{i=1}^{n} x_i = 1\f$ one can follow the following approach:
 * take \f$n-1\f$ random numbers from the interval (0,1)(0,1), then add a 0 and 1 to get a list of \f$n+1\f$ numbers.
 * Sort the list and record the differences between two consecutive elements. This creates
 * a list of \f$n\f$ number that, by construction, will sum up to 1. Moreover this sampling is uniform.
 * As an example the following code would generate points distributed on a \f$n\f$ dimensional simplex:
 *
 * @code{.unparsed}
 * std::vector<std::vector<double>> points_on_a_simplex;
 * halton ld_rng(n-1);
 * for (auto i = 0u; i < 100u; ++i) {
 *      points_on_a_simplex.push_back(project_to_simplex(ld_rng()));
 * }
 * @endcode
 *
 * @param in a <tt>std::vector</tt> containing a point in \f$n+1\f$ dimensions.
 * @return a <tt>std::vector</tt> containing the projected point of \f$n\f$ dimensions.
 *
 * @throws std::invalid_argument if the input vector elements are not in [0,1]
 * @throws std::invalid_argument if the input vector has size 0 or 1.
 *
 * See: Donald B. Rubin, The Bayesian bootstrap Ann. Statist. 9, 1981, 130-134.
 */
std::vector<double> sample_from_simplex(std::vector<double> in)
{
    if (std::any_of(in.begin(), in.end(), [](double item) { return (item < 0 || item > 1); })) {
        pagmo_throw(std::invalid_argument, "Input vector must have all elements in [0,1]");
    }
    if (in.size() > 0u) {
        std::sort(in.begin(), in.end(), detail::less_than_f<double>);
        in.insert(in.begin(), 0.0);
        in.push_back(1.0);
        for (decltype(in.size()) i = 0u; i < in.size() - 1u; ++i) {
            in[i] = in[i + 1u] - in[i];
        }
        in.pop_back();
        return in;
    } else {
        pagmo_throw(std::invalid_argument, "Input vector must have at least dimension 1, a size of "
                                               + std::to_string(in.size()) + " was detected instead.");
    }
}

van_der_corput::van_der_corput(unsigned b, unsigned n) : m_base(b), m_counter(n)
{
    if (b < 2u) {
        pagmo_throw(std::invalid_argument, "The base of the van der Corput sequence must be at least 2: "
                                               + std::to_string(b) + " was detected");
    }
}

/// Returns the next number in the sequence
/**
 * @return the next number in the sequence
 */
double van_der_corput::operator()()
{
    double retval = 0.;
    double f = 1.0 / m_base;
    unsigned i = m_counter;
    while (i > 0u) {
        retval += f * (i % m_base);
        i = i / m_base;
        f = f / m_base;
    }
    ++m_counter;
    return retval;
}

halton::halton(unsigned dim, unsigned n) : m_dim(dim)
{
    for (auto i = 0u; i < m_dim; ++i) {
        m_vdc.push_back(van_der_corput(detail::prime(i + 1), n));
    }
}

/// Returns the next number in the sequence
/**
 * @return the next number in the sequence
 */
std::vector<double> halton::operator()()
{
    std::vector<double> retval;
    for (auto i = 0u; i < m_dim; ++i) {
        retval.push_back(m_vdc[i]());
    }
    return retval;
}

} // namespace pagmo
