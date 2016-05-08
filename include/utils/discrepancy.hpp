#ifndef PAGMO_DISCREPANCY_HPP
#define PAGMO_DISCREPANCY_HPP

/** \file discrepancy.hpp
 * \brief Low-discrepancy sequences
 *
 * This header contains utilities to generate low discrepancy sequences 
 */

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

} // namespace pagmo
#endif
