#ifndef PAGMO_ALGORITHMS_COMPASS_SEARCH_HPP
#define PAGMO_ALGORITHMS_COMPASS_SEARCH_HPP

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
#if defined(DOXYGEN_INVOKED)
    /// Single entry of the log (iter, range, best)
    typedef std::tuple<unsigned int, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;
#else
    using log_line_type = std::tuple<unsigned int, double, double>;
    using log_type = std::vector<log_line_type>;
#endif

    /// Constructor
    compass_search(unsigned int max_iters = 1, double stop_range = 0.01, double start_range = 0.1,
                   double reduction_coeff = 0.5)
        : m_max_iters(max_iters), m_stop_range(stop_range), m_start_range(start_range),
          m_reduction_coeff(reduction_coeff)
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
        stream(ss, "\tMaximum number of iterations: ", m_max_iters);
        stream(ss, "\n\tStart range: ", m_start_range);
        stream(ss, "\n\tStop range: ", m_stop_range);
        stream(ss, "\n\tReduction coefficient: ", m_reduction_coeff);
        return ss.str();
    }

    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_max_iters, m_start_range, m_stop_range, m_reduction_coeff);
    }

private:
    unsigned int m_max_iters;
    double m_start_range;
    double m_stop_range;
    double m_reduction_coeff;
};

} // namespaces

PAGMO_REGISTER_ALGORITHM(pagmo::compass_search)

#endif
