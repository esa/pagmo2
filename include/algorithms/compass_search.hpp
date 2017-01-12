#ifndef PAGMO_ALGORITHMS_COMPASS_SEARCH_HPP
#define PAGMO_ALGORITHMS_COMPASS_SEARCH_HPP

#include <iomanip>
#include <stdexcept>
#include <sstream> //std::osstringstream
#include <string>
#include <vector>

#include "../algorithm.hpp"
#include "../detail/population_fwd.hpp"
#include "../exceptions.hpp"

namespace pagmo
{

/// The Compass Search Solver (CS)
/**
 *
 * \image html compass_search.png "Compass Search Illustration from Kolda et al."
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
 *
 * @see Kolda, Lewis, Torczon: 'Optimization by Direct Search: New Perspectives on Some Classical and Modern Methods'
 * published in the SIAM Journal Vol. 45, No. 3, pp. 385-482 (2003) (http://www.cs.wm.edu/~va/research/sirev.pdf)
 */
class compass_search
{
public:
    /// Single entry of the log (feval, best fitness, range)
    typedef std::tuple<unsigned long long, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    compass_search(unsigned int max_fevals = 1, double start_range = .1, double stop_range = .01,
                   double reduction_coeff = .5)
        : m_max_fevals(max_fevals), m_start_range(start_range), m_stop_range(stop_range),
          m_reduction_coeff(reduction_coeff), m_verbosity(0u), m_log()
    {
        if (start_range > 1. || start_range <= 0.) {
            pagmo_throw(std::invalid_argument, "The start range must be in (0, 1], while a value of "
                                                   + std::to_string(start_range) + " was detected.");
        }
        if (stop_range > 1. || stop_range > start_range) {
            pagmo_throw(std::invalid_argument, "the stop range must be in (start_range, 1], while a value of "
                                                   + std::to_string(stop_range) + " was detected.");
        }
        if (reduction_coeff >= 1. || reduction_coeff <= 0.) {
            pagmo_throw(std::invalid_argument, "The reduction coefficient must be in (0,1), while a value of "
                                                   + std::to_string(reduction_coeff) + " was detected.");
        }
    }

    /// Algorithm implementation
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed (pop.set_problem_seed is)
        auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto prob_f_dimension = prob.get_nf();

        auto fevals0 = prob.get_fevals(); // discount for the already made fevals
        unsigned int count = 1u;          // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this
        // particular algorithm.
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob_f_dimension != 1u) {
            pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The problem appears to be stochastic " + get_name() + " cannot deal with it");
        }
        // Get out if there is nothing to do.
        if (m_max_fevals == 0u) {
            return pop;
        }
        if (pop.size() == 0u) {
            pagmo_throw(std::invalid_argument, prob.get_name()
                                                   + " does not work on an empty population, a population size of "
                                                   + std::to_string(pop.size()) + " was, instead, detected");
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // We run the compass search starting from the best individual of the population
        auto best_idx = pop.best_idx();
        auto cur_best_x = pop.get_x()[best_idx];
        auto cur_best_f = pop.get_f()[best_idx];

        // We need some auxiliary variables
        bool flag = false;
        unsigned int fevals = 0u;

        double newrange = m_start_range;

        while (newrange > m_stop_range && fevals <= m_max_fevals) {
            flag = false;
            for (unsigned int i = 0u; i < dim; i++) {
                auto x_trial = cur_best_x;
                // move up
                x_trial[i] = cur_best_x[i] + newrange * (ub[i] - lb[i]);
                // feasibility correction
                if (x_trial[i] > ub[i]) x_trial[i] = ub[i];
                // objective function evaluation
                auto f_trial = prob.fitness(x_trial);
                fevals++;
                if (f_trial[0] < cur_best_f[0]) {
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
                if (f_trial[0] < cur_best_f[0]) {
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
                    print("\n", std::setw(7), "Fevals:", std::setw(15), "Best:", std::setw(15), "Range:", '\n');
                }
                // 2 - Print
                print(std::setw(7), prob.get_fevals() - fevals0, std::setw(15), cur_best_f[0], std::setw(15), newrange,
                      '\n');
                ++count;
                // Logs
                m_log.push_back(log_line_type(prob.get_fevals() - fevals0, cur_best_f[0], newrange));
            }
        } // end while

        if (m_verbosity) {
            if (newrange <= m_stop_range) {
                std::cout << "\nExit condition -- range: " << newrange << " <= " << m_stop_range << "\n";
            }
            else {
                std::cout << "\nExit condition -- fevals: " << fevals << " > " << m_max_fevals << "\n";
            }
        }

        // Force the current best into the original population
        pop.set_xf(best_idx, cur_best_x, cur_best_f);
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
     * @code
     * Fevals:          Best:         Range:
     *       2    1.25998e+06            0.5
     *      14    1.18725e+06            0.5
     *      28    1.18213e+06            0.5
     *      46         715894            0.5
     *      69         658715            0.5
     *     326        89696.4            0.5
     *     376        89696.4           0.25
     *     382          84952           0.25
     * @endcode
     * Fevals, is the number of function evaluations made, Best is the best fitness
     * and Range is the range used at that point of the search
     *
     * @param level verbosity level
     */
    void set_verbosity(unsigned int level)
    {
        m_verbosity = level;
    };
    /// Gets the verbosity level
    unsigned int get_verbosity() const
    {
        return m_verbosity;
    }

    /// Gets the maximum number of iterations allowed
    double get_max_fevals() const
    {
        return m_max_fevals;
    }

    /// Gets the stop_range
    double get_stop_range() const
    {
        return m_stop_range;
    }

    /// Get the start range
    double get_start_range() const
    {
        return m_start_range;
    }

    /// Get the reduction_coeff
    double get_reduction_coeff() const
    {
        return m_reduction_coeff;
    }

    /// Problem name
    std::string get_name() const
    {
        return "Compass Search";
    }

    /// Extra informations
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
     * <tt> std::vector </tt> is a compass_search::log_line_type containing: Fevals, Best, Range as described
     * in compass_search::set_verbosity
     * @return an <tt> std::vector </tt> of compass_search::log_line_type containing the logged values Fevals, Best, Range
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_max_fevals, m_start_range, m_stop_range, m_reduction_coeff, m_verbosity, m_log);
    }

private:
    unsigned int m_max_fevals;
    double m_start_range;
    double m_stop_range;
    double m_reduction_coeff;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespaces

PAGMO_REGISTER_ALGORITHM(pagmo::compass_search)

#endif
