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

#ifndef PAGMO_ALGORITHMS_SIMULATED_ANNEALING_HPP
#define PAGMO_ALGORITHMS_SIMULATED_ANNEALING_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{

/// Simulated Annealing, Corana's version with adaptive neighbourhood.
/**
 * \image html Hill_Climbing_with_Simulated_Annealing.gif
 *
 * This version of the simulated annealing algorithm is, essentially, an iterative random search
 * procedure with adaptive moves along the coordinate directions. It permits uphill moves under
 * the control of metropolis criterion, in the hope to avoid the first local minima encountered.
 *
 * The implementation provided here allows to obtain a reannealing procedure via subsequent calls
 * to the pagmo::simulated_annealing::evolve() method.
 *
 * \verbatim embed:rst:leading-asterisk
 *
 * .. warning::
 *
 *    The algorithm is not suitable for multi-objective problems, nor for
 *    constrained or stochastic optimization
 *
 * .. note::
 *
 *    When selecting the starting and final temperature values it helps to think about the tempertaure
 *    as the deterioration in the objective function value that still has a 37% chance of being accepted.
 *
 * .. note::
 *
 *    At each call of the evolve method the number of fitness evaluations will be
 *    `n_T_adj` * `n_range_adj` * `bin_size` times the problem dimension
 *
 * .. seealso::
 *
 *    Corana, A., Marchesi, M., Martini, C., & Ridella, S. (1987). Minimizing multimodal
 *    functions of continuous variables with the “simulated annealing” algorithm Corrigenda
 *    for this article is available here. ACM Transactions on Mathematical Software (TOMS), 13(3), 262-280.
 *
 * .. seealso::
 *
 *    http://people.sc.fsu.edu/~inavon/5420a/corana.pdf
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC simulated_annealing : public not_population_based
{
public:
    /// Single entry of the log (fevals, best, current, avg_range, temperature)
    typedef std::tuple<unsigned long long, double, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs simulated_annealing
     *
     * @param Ts starting temperature
     * @param Tf final temperature
     * @param n_T_adj number of temperature adjustments in the annealing schedule
     * @param n_range_adj number of adjustments of the search range performed at a constant temperature
     * @param bin_size number of mutations that are used to compute the acceptance rate
     * @param start_range starting range for mutating the decision vector
     * @param seed seed used by the internal random number generator
     *
     * @throws std::invalid_argument if \p Ts or \p Tf are not finite and positive, \p start_range is not in (0,1],
     * \p n_T_adj or n_range_adj \p are not strictly positive
     * @throws if \p Tf > \p Ts
     */
    simulated_annealing(double Ts = 10., double Tf = .1, unsigned n_T_adj = 10u, unsigned n_range_adj = 1u,
                        unsigned bin_size = 20u, double start_range = 1., unsigned seed = pagmo::random_device::next());

    // Algorithm evolve method
    population evolve(population) const;

    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >=1: will print and log one line at minimum every \p level fitness evaluations.
     *
     * Example (verbosity 5000):
     * @code{.unparsed}
     * Fevals:          Best:       Current:    Mean range:   Temperature:
     *  ...
     *  45035      0.0700823       0.135928     0.00116657      0.0199526
     *  50035      0.0215442      0.0261641    0.000770297           0.01
     *  55035     0.00551839      0.0124842    0.000559839     0.00501187
     *  60035     0.00284761     0.00703856    0.000314098     0.00251189
     *  65035     0.00264808      0.0114764    0.000314642     0.00125893
     *  70035      0.0011007     0.00293813    0.000167859    0.000630957
     *  75035    0.000435798     0.00184352    0.000126954    0.000316228
     *  80035    0.000287984    0.000825294    8.91823e-05    0.000158489
     *  85035     9.5885e-05    0.000330647    6.49981e-05    7.94328e-05
     *  90035     4.7986e-05    0.000148512    4.24692e-05    3.98107e-05
     *  95035    2.43633e-05    2.43633e-05    2.90025e-05    1.99526e-05
     * @endcode
     *
     * Fevals is the number of function evaluation used, Best is the best fitness
     * function found, Current is the last fitness sampled, Mean range is the Mean
     * search range across the decision vector components, Temperature is the current temperature.
     *
     * @param level verbosity level
     */
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    }

    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned get_verbosity() const
    {
        return m_verbosity;
    }

    // Sets the seed
    void set_seed(unsigned);

    /// Gets the seed
    /**
     * @return the seed controlling the algorithm stochastic behaviour
     */
    unsigned get_seed() const
    {
        return m_seed;
    }

    /// Algorithm name
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing the algorithm name
     */
    std::string get_name() const
    {
        return "SA: Simulated Annealing (Corana's)";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a simulated_annealing::log_line_type containing: Fevals, Best, Current, Mean range
     * Temperature as described in simulated_annealing::set_verbosity
     * @return an <tt>std::vector</tt> of simulated_annealing::log_line_type containing the logged values Gen, Fevals,
     * Best, Improvement, Mutations
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    // Object serialization
    template <typename Archive>
    void serialize(Archive &ar, unsigned);

private:
    // Starting temperature
    double m_Ts;
    // Final temperature
    double m_Tf;
    // Number of temperature adjustments during the annealing procedure
    unsigned m_n_T_adj;
    // Number of range adjustments performed at each temperature
    unsigned m_n_range_adj;
    // Number of mutation trials to evaluate the acceptance rate
    unsigned m_bin_size;
    // Starting neighbourhood size
    double m_start_range;

    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
    // Deleting the methods load save public in base as to avoid conflict with serialize
    template <typename Archive>
    void load(Archive &ar, unsigned) = delete;
    template <typename Archive>
    void save(Archive &ar, unsigned) const = delete;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::simulated_annealing)

#endif
