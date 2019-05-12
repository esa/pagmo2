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

#ifndef PAGMO_ALGORITHMS_COMPASS_SEARCH_HPP
#define PAGMO_ALGORITHMS_COMPASS_SEARCH_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>

namespace pagmo
{

/// The Compass Search Solver (CS)
/**
 * \image html compass_search.png "Compass Search Illustration from Kolda et al." width=3cm
 *
 * In the review paper by Kolda, Lewis, Torczon: 'Optimization by Direct Search: New Perspectives on Some Classical and
 * Modern Methods'
 * published in the SIAM Journal Vol. 45, No. 3, pp. 385-482 (2003), the following description of the compass search
 * algorithm is given:
 *
 * 'Davidon describes what is one of the earliest examples of a direct
 * search method used on a digital computer to solve an optimization problem:
 * Enrico Fermi and Nicholas Metropolis used one of the first digital computers,
 * the Los Alamos Maniac, to determine which values of certain theoretical
 * parameters (phase shifts) best fit experimental data (scattering cross
 * sections). They varied one theoretical parameter at a time by steps
 * of the same magnitude, and when no such increase or decrease in any one
 * parameter further improved the fit to the experimental data, they halved
 * the step size and repeated the process until the steps were deemed sufficiently
 * small. Their simple procedure was slow but sure, and several of us
 * used it on the Avidac computer at the Argonne National Laboratory for
 * adjusting six theoretical parameters to fit the pion-proton scattering data
 * we had gathered using the University of Chicago synchrocyclotron.
 * While this basic algorithm undoubtedly predates Fermi and Metropolis, it has remained
 * a standard in the scientific computing community for exactly the reason observed
 * by Davidon: it is slow but sure'.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    This algorithm does not work for multi-objective problems, nor for stochastic problems.
 *
 * .. note::
 *
 *    The search range is defined relative to the box-bounds. Hence, unbounded problems
 *    will produce an error.
 *
 * .. note::
 *
 *    Compass search is a fully deterministic algorithms and will produce identical results if its evolve method is
 *    called from two identical populations.
 *
 * .. seealso::
 *
 *    Kolda, Lewis, Torczon: 'Optimization by Direct Search: New Perspectives on Some Classical and Modern Methods'
 *    published in the SIAM Journal Vol. 45, No. 3, pp. 385-482 (2003)
 *
 *    http://www.cs.wm.edu/~va/research/sirev.pdf
 *
 * \endverbatim
 *
 */
class PAGMO_DLL_PUBLIC compass_search : public not_population_based
{
public:
    /// Single entry of the log (feval, best fitness, n. constraints violated, violation norm, range)
    typedef std::tuple<unsigned long long, double, vector_double::size_type, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor.
    /**
     * Constructs compass_search
     *
     * @param max_fevals maximum number of fitness evaluations
     * @param start_range start range
     * @param stop_range stop range
     * @param reduction_coeff range reduction coefficient
     * @throws std::invalid_argument if \p start_range is not in (0,1]
     * @throws std::invalid_argument if \p stop_range is not in (start_range,1]
     * @throws std::invalid_argument if \p reduction_coeff is not in (0,1)
     */
    compass_search(unsigned max_fevals = 1, double start_range = .1, double stop_range = .01,
                   double reduction_coeff = .5);

    // Algorithm evolve method (juice implementation of the algorithm)
    population evolve(population) const;

    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >0: will print and log one line each objective function improvement, or range reduction
     *
     * Example (verbosity > 0u):
     * @code{.unparsed}
     * Fevals:          Best:      Violated:    Viol. Norm:         Range:
     *       4        110.785              1        2.40583            0.5
     *      12        110.785              1        2.40583           0.25
     *      20        110.785              1        2.40583          0.125
     *      22        91.0454              1        1.01855          0.125
     *      25        96.2795              1       0.229446          0.125
     *      33        96.2795              1       0.229446         0.0625
     *      41        96.2795              1       0.229446        0.03125
     *      ..        .......              .       ........        0.03125
     *     111        95.4617              1     0.00117433    0.000244141
     *     115        95.4515              0              0    0.000244141
     *     123        95.4515              0              0     0.00012207
     *     131        95.4515              0              0    6.10352e-05
     *     139        95.4515              0              0    3.05176e-05
     *     143        95.4502              0              0    3.05176e-05
     *     151        95.4502              0              0    1.52588e-05
     *     159        95.4502              0              0    7.62939e-06
     * Exit condition -- range: 7.62939e-06 <= 1e-05
     * @endcode
     * Fevals, is the number of fitness evaluations made, Best is the best fitness
     * Violated and Viol.Norm are the number of constraints violated and the L2 norm of the violation (accounting for
     the
     * tolerances returned by problem::get_c_tol, and Range is the range used at that point of the search
     *
     * @param level verbosity level
     */
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    };
    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned get_verbosity() const
    {
        return m_verbosity;
    }
    /// Gets the maximum number of iterations allowed
    /**
     * @return the maximum number of iterations allowed
     */
    double get_max_fevals() const
    {
        return m_max_fevals;
    }
    /// Gets the stop_range
    /**
     * @return the stop range
     */
    double get_stop_range() const
    {
        return m_stop_range;
    }
    /// Get the start range
    /**
     * @return the start range
     */
    double get_start_range() const
    {
        return m_start_range;
    }
    /// Get the reduction_coeff
    /**
     * @return the reduction coefficient
     */
    double get_reduction_coeff() const
    {
        return m_reduction_coeff;
    }
    /// Algorithm name
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing the algorithm name
     */
    std::string get_name() const
    {
        return "CS: Compass Search";
    }
    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a compass_search::log_line_type containing: Fevals, Best, Violated and Viol.Norm,
     * Range as described in compass_search::set_verbosity
     * @return an <tt>std::vector</tt> of compass_search::log_line_type containing the logged values Fevals, Best,
     * Range
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    unsigned m_max_fevals;
    double m_start_range;
    double m_stop_range;
    double m_reduction_coeff;
    unsigned m_verbosity;
    mutable log_type m_log;
    // Deleting the methods load save public in base as to avoid conflict with serialize
    template <typename Archive>
    void load(Archive &, unsigned) = delete;
    template <typename Archive>
    void save(Archive &, unsigned) const = delete;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::compass_search)

#endif
