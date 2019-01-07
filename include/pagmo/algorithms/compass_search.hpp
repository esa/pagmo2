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

#include <cmath> //std::isnan
#include <iomanip>
#include <sstream> //std::osstringstream
#include <stdexcept>
#include <string>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/utils/constrained.hpp>

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
class compass_search : public not_population_based
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
    compass_search(unsigned int max_fevals = 1, double start_range = .1, double stop_range = .01,
                   double reduction_coeff = .5)
        : m_max_fevals(max_fevals), m_start_range(start_range), m_stop_range(stop_range),
          m_reduction_coeff(reduction_coeff), m_verbosity(0u), m_log()
    {
        if (start_range > 1. || start_range <= 0. || std::isnan(start_range)) {
            pagmo_throw(std::invalid_argument, "The start range must be in (0, 1], while a value of "
                                                   + std::to_string(start_range) + " was detected.");
        }
        if (stop_range > 1. || stop_range >= start_range || std::isnan(stop_range)) {
            pagmo_throw(std::invalid_argument, "the stop range must be in (start_range, 1], while a value of "
                                                   + std::to_string(stop_range) + " was detected.");
        }
        if (reduction_coeff >= 1. || reduction_coeff <= 0. || std::isnan(reduction_coeff)) {
            pagmo_throw(std::invalid_argument, "The reduction coefficient must be in (0,1), while a value of "
                                                   + std::to_string(reduction_coeff) + " was detected.");
        }
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     * Evolves the population up to when the search range becomes smaller than
     * the defined stop_range
     *
     * @param pop population to be evolved
     * @return evolved population
     * @throws std::invalid_argument if the problem is multi-objective or stochastic
     * @throws std::invalid_argument if the population is empty
     */
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed
        auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto neq = prob.get_nec();

        auto fevals0 = prob.get_fevals(); // discount for the already made fevals
        unsigned int count = 1u;          // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this
        // particular algorithm.
        if (prob.get_nobj() != 1u) {
            pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The problem appears to be stochastic " + get_name() + " cannot deal with it");
        }
        if (pop.size() == 0u) {
            pagmo_throw(std::invalid_argument, get_name() + " does not work on an empty population");
        }
        // Get out if there is nothing to do.
        if (m_max_fevals == 0u) {
            return pop;
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // We init the starting point
        auto sel_xf = select_individual(pop);
        vector_double cur_best_x(std::move(sel_xf.first)), cur_best_f(std::move(sel_xf.second));

        // We need some auxiliary variables
        bool flag = false;
        unsigned int fevals = 0u;

        double newrange = m_start_range;

        while (newrange > m_stop_range && fevals <= m_max_fevals) {
            flag = false;
            for (decltype(dim) i = 0u; i < dim; i++) {
                auto x_trial = cur_best_x;
                // move up
                x_trial[i] = cur_best_x[i] + newrange * (ub[i] - lb[i]);
                // feasibility correction
                if (x_trial[i] > ub[i]) x_trial[i] = ub[i];
                // objective function evaluation
                auto f_trial = prob.fitness(x_trial);
                fevals++;
                if (compare_fc(f_trial, cur_best_f, prob.get_nec(), prob.get_c_tol())) {
                    cur_best_f = f_trial;
                    cur_best_x = x_trial;
                    flag = true;
                    break; // accept
                }

                // move down
                x_trial[i] = cur_best_x[i] - newrange * (ub[i] - lb[i]);
                // feasibility correction
                if (x_trial[i] < lb[i]) x_trial[i] = lb[i];
                // objective function evaluation
                f_trial = prob.fitness(x_trial);
                fevals++;
                if (compare_fc(f_trial, cur_best_f, prob.get_nec(), prob.get_c_tol())) {
                    cur_best_f = f_trial;
                    cur_best_x = x_trial;
                    flag = true;
                    break; // accept
                }
            }
            if (!flag) {
                newrange *= m_reduction_coeff;
            }

            // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Prints a log line if a new best is found or the range has been decreased

                // 1 - Every 50 lines print the column names
                if (count % 50u == 1u) {
                    print("\n", std::setw(7), "Fevals:", std::setw(15), "Best:", std::setw(15),
                          "Violated:", std::setw(15), "Viol. Norm:", std::setw(15), "Range:", '\n');
                }
                // 2 - Print
                auto c1eq = detail::test_eq_constraints(cur_best_f.data() + 1, cur_best_f.data() + 1 + neq,
                                                        prob.get_c_tol().data());
                auto c1ineq = detail::test_ineq_constraints(
                    cur_best_f.data() + 1 + neq, cur_best_f.data() + cur_best_f.size(), prob.get_c_tol().data() + neq);
                auto n = prob.get_nc() - c1eq.first - c1ineq.first;
                auto l = c1eq.second + c1ineq.second;
                print(std::setw(7), prob.get_fevals() - fevals0, std::setw(15), cur_best_f[0], std::setw(15), n,
                      std::setw(15), l, std::setw(15), newrange, '\n');
                ++count;
                // Logs
                m_log.emplace_back(prob.get_fevals() - fevals0, cur_best_f[0], n, l, newrange);
            }
        } // end while

        if (m_verbosity) {
            if (newrange <= m_stop_range) {
                std::cout << "Exit condition -- range: " << newrange << " <= " << m_stop_range << "\n";
            } else {
                std::cout << "Exit condition -- fevals: " << fevals << " > " << m_max_fevals << "\n";
            }
        }

        // Force the current best into the original population
        replace_individual(pop, cur_best_x, cur_best_f);
        return pop;
    };

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
    void set_verbosity(unsigned int level)
    {
        m_verbosity = level;
    };
    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned int get_verbosity() const
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
    /// Extra informations
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tMaximum number of objective function evaluations: ", m_max_fevals);
        stream(ss, "\n\tStart range: ", m_start_range);
        stream(ss, "\n\tStop range: ", m_stop_range);
        stream(ss, "\n\tReduction coefficient: ", m_reduction_coeff);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        return ss.str();
    }

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

    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<not_population_based>(this), m_max_fevals, m_start_range, m_stop_range, m_reduction_coeff,
           m_verbosity, m_log);
    }

private:
    unsigned int m_max_fevals;
    double m_start_range;
    double m_stop_range;
    double m_reduction_coeff;
    unsigned int m_verbosity;
    mutable log_type m_log;
    // Deleting the methods load save public in base as to avoid conflict with serialize
    template <typename Archive>
    void load(Archive &ar) = delete;
    template <typename Archive>
    void save(Archive &ar) const = delete;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::compass_search)

#endif
