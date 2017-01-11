#ifndef PAGMO_ALGORITHMS_COMPASS_SEARCH_HPP
#define PAGMO_ALGORITHMS_COMPASS_SEARCH_HPP

#include <exceptions>
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
    /// Single entry of the log (feval, range, best fitness)
    typedef std::tuple<unsigned int, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    compass_search(unsigned int max_fevals = 1, double stop_range = 0.01, double start_range = 0.1,
                   double reduction_coeff = 0.5)
        : m_max_fevals(max_fevals), m_stop_range(stop_range), m_start_range(start_range),
          m_reduction_coeff(reduction_coeff), m_verbosity(0u), m_log()
    {
        if (reduction_coeff >= 1. || reduction_coeff <= 0.) {
            pagmo_throw(std::invalid_argument, "The reduction coefficient must be in (0,1), while a value of "
                                                   + std::to_string(reduction_coeff) + " was detected.");
        }
        if (start_range > 1. || start_range <= 0.) {
            pagmo_throw(std::invalid_argument, "The start range must be in (0, 1], while a value of "
                                                   + std::to_string(start_range) + " was detected.");
        }
        if (stop_range > 1. || stop_range > start_range) {
            pagmo_throw(std::invalid_argument, "the stop range must be in (start_range, 1], while a value of "
                                                   + std::to_string(stop_range) + " was detected.");
        }
    }

    /// Algorithm implementation
    population evolve(const population &pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed (pop.set_problem_seed is)
        auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto prob_f_dimension = prob.get_nf();

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
        if (max_iters == 0u) {
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
            for (unsigned int i = 0u; i < Dc; i++) {
                auto x_trial = cur_best_x;
                // move up
                x_trial[i] = cur_best_x[i] + newrange * (ub[i] - lb[i]);
                // feasibility correction
                if (x_trial[i] > ub[i]) x_trial[i] = ub[i];
                // objective function evaluation
                auto f_trial = prob.objfun(x_trial);
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
                f_trial = prob.objfun(x_trial);
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
        } // end while
        // Force the current best into the original population
        pop.set_xf(best_idx, cur_best_f, cur_best_x);
        return pop;
    };

    /// Gets the maximum number of iterations allowed
    unsigned int get_max_iters() const
    {
        return m_max_iters;
    }

    /// Gets the stop_range
    unsigned int get_stop_range() const
    {
        return m_stop_range;
    }

    /// Get the start range
    unsigned int get_start_range() const
    {
        return m_start_range;
    }

    /// Get the reduction_coeff
    unsigned int get_reduction_coeff() const
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

    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_max_iters, m_start_range, m_stop_range, m_reduction_coeff, m_verbosity, m_log);
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
