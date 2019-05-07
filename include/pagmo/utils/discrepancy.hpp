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

#ifndef PAGMO_DISCREPANCY_HPP
#define PAGMO_DISCREPANCY_HPP

#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/s11n.hpp>

namespace pagmo
{

// Sample from a simplex
PAGMO_DLL_PUBLIC std::vector<double> sample_from_simplex(std::vector<double>);

/// Van der Corput sequence
/**
 * A Van der Corput sequence is the simplest one-dimensional low-discrepancy sequence over the
 * unit interval; it was first described in 1935 by the Dutch mathematician Johannes van der Corput.
 * It is constructed by reversing the base representation of the sequence of natural number (1, 2, 3, …).
 * A positive integer \f$n \ge 1\f$ is represented, in the base \f$b\f$ by:
 * \f[
 * n = \sum_{i=0}^{L-1}d_i(n) b^i,
 * \f]
 * where \f$L\f$ is the number of digits needed.
 * The \f$n\f$-th number in a van der Corput sequence is thus defined as:
 * \f[
 * g_n=\sum_{i=0}^{L-1}d_i(n) b^{-i-1}.
 * \f]
 *
 * so that, for example, if \f$b = 10\f$:
 *
 * \f$ seq = \{ 0, \tfrac{1}{10}, \tfrac{2}{10}, \tfrac{3}{10}, \tfrac{4}{10}, \tfrac{5}{10}, \tfrac{6}{10},
 * \tfrac{7}{10}, \tfrac{8}{10}, \tfrac{9}{10}, \tfrac{1}{100}, \tfrac{11}{100}, \tfrac{21}{100},
 * \tfrac{31}{100}, \tfrac{41}{100}, \tfrac{51}{100}, \tfrac{61}{100}, \tfrac{71}{100}, \tfrac{81}{100},
 * \tfrac{91}{100}, \tfrac{2}{100}, \tfrac{12}{100}, \tfrac{22}{100}, \tfrac{32}{100}, \ldots \} \,\f$
 *
 * or, if \f$b = 2\f$:
 *
 * \f$ seq = \{0, \tfrac{1}{2}, \tfrac{1}{4}, \tfrac{3}{4}, \tfrac{1}{8}, \tfrac{5}{8}, \tfrac{3}{8},
 * \tfrac{7}{8}, \tfrac{1}{16}, \tfrac{9}{16}, \tfrac{5}{16}, \tfrac{13}{16}, \tfrac{3}{16}, \tfrac{11}{16},
 * \tfrac{7}{16}, \tfrac{15}{16}, \ldots.\} \f$
 *
 * See: https://en.wikipedia.org/wiki/Van_der_Corput_sequence
 *
 */
class PAGMO_DLL_PUBLIC van_der_corput
{
public:
    /// Constructor from base and starting element
    /**
     * Consruct a van der Corput lowp-discrepancy sequence with base
     * \p b and starting element position \p n
     *
     * @param b base
     * @param n position of the starting element
     *
     * @throws std::invalid_argument if the base is 0u or 1u
     */
    van_der_corput(unsigned b = 2u, unsigned n = 0u);
    // Returns the next number in the sequence
    double operator()();
    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        detail::archive(ar, m_base, m_counter);
    }

private:
    // Base of the sequence
    unsigned m_base;
    // Element of the sequence to compute
    unsigned m_counter;
};

/// Halton sequence
/**
 * The Halton sequence is, essentially, a generalization of the van der Corput sequence
 * to higher dimensions. It considers, along each dimension, a van der Corput sequence
 * referred to co-prime numbers. Here, by default, we consider the sequence of all prime
 * numbers starting from 2, 3, 5, ...... so that, for example, for \p dim equal two the
 * following sequence is generated:
 *
 * \f[
 * seq = \left\{ (0, 0), \left(\frac 12, \frac 13\right), \left(\frac 14, \frac 23\right), \left(\frac 34, \frac
 * 19\right), \left(\frac 18,
 * \frac 49\right), \left(\frac 58, \frac 79\right), \left(\frac 38, \frac 29\right), ... \right\}
 * \f]
 *
 * @param n selects which element of the sequence to return
 * @param dim dimensions of the returned point
 *
 * See: https://en.wikipedia.org/wiki/Halton_sequence
 *
 */
class PAGMO_DLL_PUBLIC halton
{
public:
    /// Constructor from base and starting element
    /**
     * Consruct a Halton low-discrepancy sequence with dimension
     * \p dim and starting element position \p n
     *
     * @param dim dimension
     * @param n position of the starting element
     *
     * @throws unspecified all exceptions thrown by pagmo::van_der_corput
     */
    halton(unsigned dim = 2u, unsigned n = 0u);
    // Returns the next number in the sequence
    std::vector<double> operator()();
    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of pagmo::van_der_corput or of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        detail::archive(ar, m_dim, m_vdc);
    }

private:
    // Dimension of the sequence
    unsigned m_dim;
    // van der Corput sequences used for each dimension
    std::vector<van_der_corput> m_vdc;
};

} // namespace pagmo

#endif
