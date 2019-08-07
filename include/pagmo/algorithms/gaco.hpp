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

#ifndef PAGMO_ALGORITHMS_GACO_HPP
#define PAGMO_ALGORITHMS_GACO_HPP

#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <boost/optional.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{
/// Extended Ant Colony Opitmization
/**
 * \image html gaco.png "Ant Colony Optimization Illustration" width=0.5cm
 *
 * Ant colony optimization is a class of optimization algorithms modeled on the actions
 * of an ant colony. Artificial 'ants' (e.g. simulation agents) locate optimal solutions by
 * moving through a parameter space representing all possible solutions. Real ants lay down
 * pheromones directing each other to resources while exploring their environment.
 * The simulated 'ants' similarly record their positions and the quality of their solutions,
 * so that in later simulation iterations more ants locate better solutions.
 *
 * In pagmo we propose a version of this algorithm called extended ACO and originally described
 * by Schlueter et al.
 * Extended ACO generates future generations of ants by using the a multi-kernel gaussian distribution
 * based on three parameters (i.e., pheromone values) which are computed depending on the quality
 * of each previous solution. The solutions are ranked through an oracle penalty method.
 *
 * This algorithm can be applied to box-bounded single-objective, constrained and unconstrained
 * optimization, with both continuous and integer variables.
 *
 *
 * \verbatim embed:rst:leading-asterisk
 * .. seealso::
 *
 *    M. Schlueter, et al. (2009). Extended ant colony optimization for non-convex mixed integer non-linear programming.
 *    Computers & Operations Research.
 *
 * .. note::
 *
 *    The ACO version implemented in PaGMO is an extension of Schlueter's originally proposed extended ACO algorithm.
 *    The main difference between the implemented version  and the original one lies in
 *    how two of the three pheromone values are computed (in particular, the weights and the standard deviations).
 *
 *    Image credit: https://commons.wikimedia.org/wiki/File:Knapsack_ants.svg
 *
 * \endverbatim
 *
 */
class PAGMO_DLL_PUBLIC gaco
{
public:
    /// Single entry of the log (gen, m_fevals, best_fit, m_ker, m_oracle, dx, dp)
    typedef std::tuple<unsigned, vector_double::size_type, double, unsigned, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /**
     * Constructs the ACO user defined algorithm for single-objective optimization.
     *
     * @param[in] gen Generations: number of generations to evolve.
     * @param[in] ker Kernel: number of solutions stored in the solution archive.
     * @param[in] q Convergence speed parameter: this parameter is useful for managing the convergence speed towards the
     * found minima (the smaller the faster).
     * @param[in] oracle Oracle parameter: this is the oracle parameter used in the penalty method.
     * @param[in] acc Accuracy parameter: for maintaining a minimum penalty function's values distances.
     * @param[in] threshold Threshold parameter: when the generations reach the threshold then q is set to
     * 0.01 automatically.
     * @param[in] n_gen_mark Standard deviations convergence speed parameter: this parameters determines the convergence
     * speed of the standard deviations values.
     * @param[in] impstop Improvement stopping criterion: if a positive integer is assigned here, the algorithm will
     * count the runs without improvements, if this number will exceed impstop value, the algorithm will be stopped.
     * @param[in] evalstop Evaluation stopping criterion: same as previous one, but with function evaluations.
     * @param[in] focus Focus parameter: this parameter makes the search for the optimum greedier and more focused on
     * local improvements (the higher the greedier). If the value is very high, the search is more focused around the
     * current best solutions.
     * @param[in] memory Memory parameter: if true, memory is activated in the algorithm for multiple calls
     * @param seed seed used by the internal random number generator (default is random).
     *
     * @throws std::invalid_argument if \p acc is not \f$ >=0 \f$, \p impstop is not a
     * positive integer, \p evalstop is not a positive integer, \p focus is not \f$ >=0 \f$,
     * \p ker is not a positive integer, \p oracle is not positive, \p
     * threshold is not \f$ \in [1,gen] \f$ when \f$memory=false\f$ and  \f$gen!=0\f$, \p threshold is not \f$ >=1 \f$
     * when \f$memory=true\f$ and \f$gen!=0\f$, \p q is not \f$ >=0 \f$
     */
    gaco(unsigned gen = 1u, unsigned ker = 63u, double q = 1.0, double oracle = 0., double acc = 0.01,
         unsigned threshold = 1u, unsigned n_gen_mark = 7u, unsigned impstop = 100000u, unsigned evalstop = 100000u,
         double focus = 0., bool memory = false, unsigned seed = pagmo::random_device::next());

    // Algorithm evolve method
    population evolve(population) const;

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

    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >0: will print and log one line each \p level generations.
     *
     * Example (verbosity 1):
     * @code{.unparsed}
     *Gen:        Fevals:          Best:        Kernel:        Oracle:            dx:            dp:
     *  1              0        179.464             13            100        4.33793          47876
     *  2             15         14.205             13            100        5.20084        5928.12
     *  3             30         14.205             13         14.205        1.24173        1037.44
     *  4             45         14.205             13         14.205        3.05807         395.89
     *  5             60        7.91087             13         14.205       0.711446        286.599
     *  6             75        2.81437             13        7.91087        5.80451        71.8174
     *  7             90        2.81437             13        2.81437        1.90561        48.3829
     *  8            105        2.81437             13        2.81437         1.3072        26.9496
     *  9            120         1.4161             13        2.81437        1.61732        10.6527
     * 10            150         1.4161             13         1.4161        2.54262        3.67034
     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used, Best is the best fitness function
     *value found until that generation, Kernel is the kernel size, Oracle is the oracle parameter value, dx is the
     *flatness in the individuals, dp is the flatness in the penalty function values.
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

    /// Gets the generations
    /**
     * @return the number of generations to evolve for
     */
    unsigned get_gen() const
    {
        return m_gen;
    }

    // Sets the bfe
    void set_bfe(const bfe &b);

    /// Algorithm name
    /**
     * Returns the name of the algorithm.
     *
     * @return <tt> std::string </tt> containing the algorithm name
     */
    std::string get_name() const
    {
        return "GACO: Ant Colony Optimization";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a gaco::log_line_type containing: gen, m_fevals, best_fit, m_ker,
     * m_oracle, dx, dp
     * as described in gaco::set_verbosity
     * @return an <tt>std::vector</tt> of gaco::log_line_type containing the logged values gen, m_fevals,
     * best_fit, m_ker, m_oracle, dx, dp
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    PAGMO_DLL_LOCAL double penalty_computation(const vector_double &f, const population &pop,
                                               const unsigned long long nobj, const unsigned long long nec,
                                               const unsigned long long nic) const;
    PAGMO_DLL_LOCAL void update_sol_archive(const population &pop, vector_double &sorted_vector,
                                            std::vector<decltype(sorted_vector.size())> &sorted_list,
                                            std::vector<vector_double> &sol_archive) const;
    PAGMO_DLL_LOCAL void pheromone_computation(const unsigned gen, vector_double &prob_cumulative,
                                               vector_double &omega_vec, vector_double &sigma_vec,
                                               const population &popul, std::vector<vector_double> &sol_archive) const;
    PAGMO_DLL_LOCAL void generate_new_ants(const population &pop, std::uniform_real_distribution<> dist,
                                           std::normal_distribution<double> gauss_pdf, vector_double prob_cumulative,
                                           vector_double sigma, std::vector<vector_double> &dvs_new,
                                           std::vector<vector_double> &sol_archive) const;

    unsigned m_gen;
    double m_acc;
    unsigned m_impstop;
    unsigned m_evalstop;
    double m_focus;
    unsigned m_ker;
    mutable double m_oracle;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
    mutable double m_res;
    unsigned m_threshold;
    mutable double m_q;
    unsigned m_n_gen_mark;
    bool m_memory;
    mutable unsigned m_counter;
    mutable std::vector<vector_double> m_sol_archive;
    mutable unsigned m_n_evalstop;
    mutable unsigned m_n_impstop;
    mutable unsigned m_gen_mark;
    mutable vector_double::size_type m_fevals;
    boost::optional<bfe> m_bfe;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::gaco)

#endif
