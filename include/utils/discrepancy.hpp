#ifndef PAGMO_DISCREPANCY_HPP
#define PAGMO_DISCREPANCY_HPP

/** \file discrepancy.hpp
 * \brief Low-discrepancy sequences
 *
 * This header contains utilities to generate low discrepancy sequences 
 */

#include <algorithm>
#include <vector>

#include "../exceptions.hpp"
#include "../io.hpp"
#include "../types.hpp"


namespace pagmo{

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
 * \f$ seq = \{ \tfrac{1}{10}, \tfrac{2}{10}, \tfrac{3}{10}, \tfrac{4}{10}, \tfrac{5}{10}, \tfrac{6}{10},
 * \tfrac{7}{10}, \tfrac{8}{10}, \tfrac{9}{10}, \tfrac{1}{100}, \tfrac{11}{100}, \tfrac{21}{100},
 * \tfrac{31}{100}, \tfrac{41}{100}, \tfrac{51}{100}, \tfrac{61}{100}, \tfrac{71}{100}, \tfrac{81}{100},
 * \tfrac{91}{100}, \tfrac{2}{100}, \tfrac{12}{100}, \tfrac{22}{100}, \tfrac{32}{100}, \ldots \} \,\f$
 *
 * or, if \f$b = 2\f$:
 *
 * \f$ seq = \{\tfrac{1}{2}, \tfrac{1}{4}, \tfrac{3}{4}, \tfrac{1}{8}, \tfrac{5}{8}, \tfrac{3}{8},
 * \tfrac{7}{8}, \tfrac{1}{16}, \tfrac{9}{16}, \tfrac{5}{16}, \tfrac{13}{16}, \tfrac{3}{16}, \tfrac{11}{16},
 * \tfrac{7}{16}, \tfrac{15}{16}, \ldots.\} \f$
 *
 *
 * @param[in] n selects which number of the sequence to return
 * @param[in] b number to be used as a base of the sequence
 * @returns the n-th number in the van_der_corput sequence
 *
 * @see http://en.wikipedia.org/wiki/Van_der_Corput_sequence
 */
double van_der_corput(unsigned int n, unsigned int b) {
    double retval = 0.;
    double f = 1.0 / b;
    unsigned int i = n;
    while (i > 0) {
        retval += f * (i % b);
        i = i / b;
        f = f / b;
    }
    return retval;
}

/// Projects a point onto a simplex
/**
 * Projects a point \f$\mathbf x \in [0,1]^n\f$ onto a simplex so that if points comes from a uniform distribution
 * their projection will also be uniformly distributed on the simplex.
 *
 * In order to generate a uniform distribution on a simplex, that is to sample a \f$n\f$-dimensional
 * point \f$\mathbf x\f$ such that \f$\sum_{i=1}^{n} x_i = 1\f$ one can follow the following approach:
 * take \f$n-1\f$ random numbers from the interval (0,1)(0,1), then add a 0 and 1 to get a list of \f$n+1\f$ numbers.
 * Sort the list and record the differences between two consecutive elements. This creates
 * a list of \f$n\f$ number that, by construction, will sum up to 1. Moreover this sampling is uniform. 
 * As an example the following code would generate points uniformly distributed on a simplex:
 *
 * @code
 * std::vector<std::vector<double>> points_on_a_simplex;
 * for (auto i = 0u; i < 100u; ++i) {
 *      auto v = random_vector(n+1); \\ we assume random_vector returns a uniformly distributed random vector of size n+1
 *      points_on_a_simplex.push_back(project_2_simplex(v));
 * }
 * @endcode
 *
 * @param[in] in a <tt>std::vector</tt> containing a point in \f$n+1\f$ dimensions.
 * @return a <tt>std::vector</tt> containing the projected point of \f$n\f$ dimensions.
 *
 * @throws std::invalid_argument if the input vector elements are not in [0,1]
 * @throws std::invalid_argument if the input vector has size 0 or 1.
 *
 * @see Donald B. Rubin, The Bayesian bootstrap Ann. Statist. 9, 1981, 130-134.
 */
std::vector<double> project_2_simplex(std::vector<double> in) const
{
    if (std::any_of(in.begin(), in.end(), [](auto item){return (item < 0 || item > 1);})) {
        pagmo_throw(std::invlaid_argument,"Input vector must have all elements in [0,1]");
    }
    if (in.size() > 1) {
        std::sort(in.begin(),in.end());
        in.insert(in.begin(),0.0);
        in.push_back(1.0);
        double cumsum=0;
        for (unsigned int i = 0u; i < in.size()-1u; ++i) {
            in[i] = in[i+1] - in[i];
            cumsum += in[i];
        }
        in.pop_back();
        for (unsigned int i = 0u; i<in.size();++i) {
            in[i] /= cumsum;
        }
        return in;
    }
    else {
        pagmo_throw(std::invlaid_argument,"Input vector must have at least dimension 2, a size of " + std::to_string(in.size()) + " was detected instead.");
    }
}

} // namespace pagmo
#endif
