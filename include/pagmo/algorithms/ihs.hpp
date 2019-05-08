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

#ifndef PAGMO_ALGORITHMS_IHS_HPP
#define PAGMO_ALGORITHMS_IHS_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// Improved Harmony Search
/**
 * \image html ihs.gif
 *
 * Harmony search (HS) is a metaheuristic algorithm said to mimick the improvisation process of musicians.
 * In the metaphor, each musician (i.e., each variable) plays (i.e., generates) a note (i.e., a value)
 * for finding a best harmony (i.e., the global optimum) all together.
 *
 * This code implements the so-called improved harmony search algorithm (IHS), in which the probability
 * of picking the variables from the decision vector and the amount of mutation to which they are subject
 * vary (respectively linearly and exponentially) at each call of the ``evolve()`` method.
 *
 * In this algorithm the number of fitness function evaluations is equal to the number of iterations.
 * All the individuals in the input population participate in the evolution. A new individual is generated
 * at every iteration, substituting the current worst individual of the population if better.
 **
 *
 * \verbatim embed:rst:leading-asterisk
 *
 * .. warning::
 *
 *    The HS algorithm can and has been  criticized, not for its performances,
 *    but for the use of a metaphor that does not add anything to existing ones. The HS
 *    algorithm essentially applies mutation and crossover operators to a background population and as such
 *    should have been developed in the context of Evolutionary Strategies or Genetic Algorithms and studied
 *    in that context. The use of the musicians metaphor only obscures its internal functioning
 *    making theoretical results from ES and GA erroneously seem as unapplicable to HS.
 *
 * .. note::
 *
 *    The original IHS algorithm was designed to solve unconstrained, deterministic single objective problems.
 *    In pagmo, the algorithm was modified to tackle also multi-objective (unconstrained), constrained
 *    (single-objective), mixed-integer and stochastic problems. Such extension is original with pagmo.
 *
 * .. seealso::
 *
 *    https://en.wikipedia.org/wiki/Harmony_search for an introduction on harmony search.
 *
 * .. seealso::
 *
 *    https://linkinghub.elsevier.com/retrieve/pii/S0096300306015098 for the paper that introduces and explains improved
 *    harmony search.
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC ihs
{
public:
    /// Single data line for the algorithm's log.
    /**
     * A log data line is a tuple consisting of:
     * - the number of objective function evaluations made so far,
     * - the pitch adjustment rate,
     * - the distance bandwidth
     * - the population flatness evaluated as the distance between the decisions vector of the best and of the worst
     * individual (or -1 in a multiobjective case),
     * - the population flatness evaluated as the distance between the fitness of the best and of the worst individual
     * (or -1 in a multiobjective case),
     * - the number of constraints violated by the current decision vector,
     * - the constraints violation norm for the current decision vector,
     * - the objective value of the best solution or, in the multiobjective case, the ideal point
     */
    typedef std::tuple<unsigned long long, double, double, double, double, vector_double::size_type, double,
                       vector_double>
        log_line_type;
    /// Log type.
    /**
     * The algorithm log is a collection of ihs::log_line_type data lines, stored in chronological order
     * during the optimisation if the verbosity of the algorithm is set to a nonzero value
     * (see ihs::set_verbosity()).
     */
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs ihs
     *
     * @param gen Number of generations to consider. Each generation will compute the objective function once.
     * @param phmcr probability of choosing from memory (similar to a crossover probability)
     * @param ppar_min minimum pitch adjustment rate. (similar to a mutation rate)
     * @param ppar_max maximum pitch adjustment rate. (similar to a mutation rate)
     * @param bw_min minimum distance bandwidth. (similar to a mutation width)
     * @param bw_max maximum distance bandwidth. (similar to a mutation width)
     * @param seed seed used by the internal random number generator
     *
     * @throws value_error if phmcr is not in the ]0,1[ interval, ppar min/max are not in the ]0,1[
     * interval, min/max quantities are less than/greater than max/min quantities, bw_min is negative.
     */
    ihs(unsigned gen = 1u, double phmcr = 0.85, double ppar_min = 0.35, double ppar_max = 0.99, double bw_min = 1E-5,
        double bw_max = 1., unsigned seed = pagmo::random_device::next());

    // Algorithm evolve method
    population evolve(population) const;

    /// Set verbosity.
    /**
     * This method will set the algorithm's verbosity. If \p n is zero, no output is produced during the optimisation
     * and no logging is performed. If \p n is nonzero, then every \p n objective function evaluations the status
     * of the optimisation will be both printed to screen and recorded internally. See ihs::log_line_type and
     * ihs::log_type for information on the logging format. The internal log can be fetched via get_log().
     *
     * Example (verbosity 100, a constrained problem):
     * @code{.unparsed}
     * Fevals:          ppar:            bw:            dx:            df:      Violated:    Viol. Norm:        ideal1:
     *       1        0.35064       0.988553        5.17002        68.4027              1      0.0495288        85.1946
     *     101        0.41464       0.312608        4.21626         46.248              1      0.0495288        85.1946
     *     201        0.47864      0.0988553        2.27851        8.00679              1      0.0495288        85.1946
     *     301        0.54264      0.0312608        3.94453        31.9834              1      0.0495288        85.1946
     *     401        0.60664     0.00988553        4.74834         40.188              1      0.0495288        85.1946
     *     501        0.67064     0.00312608        2.91583        6.53575              1     0.00904482        90.3601
     *     601        0.73464    0.000988553        2.98691        10.6425              1    0.000760728        110.121
     *     701        0.79864    0.000312608        2.27775        39.7507              1    0.000760728        110.121
     *     801        0.86264    9.88553e-05       0.265908         4.5488              1    0.000760728        110.121
     *     901        0.92664    3.12608e-05       0.566348       0.354253              1    0.000760728        110.121
     * @endcode
     * Feasibility is checked against the problem's tolerance.
     *
     * By default, the verbosity level is zero.
     *
     * @param n the desired verbosity level.
     */
    void set_verbosity(unsigned n)
    {
        m_verbosity = n;
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
        return "IHS: Improved Harmony Search";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get the optimisation log.
    /**
     * See ihs::log_type for a description of the optimisation log. Logging is turned on/off via
     * set_verbosity().
     *
     * @return a const reference to the log.
     */
    const log_type &get_log() const
    {
        return m_log;
    }
    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    // logging is complex fir ihs as the algorithm is an "any-problem" wannabe
    PAGMO_DLL_LOCAL void log_a_line(const population &, unsigned &, unsigned long long, double, double) const;

    unsigned m_gen;
    double m_phmcr;
    double m_ppar_min;
    double m_ppar_max;
    double m_bw_min;
    double m_bw_max;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::ihs)

#endif
