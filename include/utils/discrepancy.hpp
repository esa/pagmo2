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
 * A positive integer \f$n \ge 1\f$ is represented, in the base \f$b\f$ and using \f$ L\f$ digits:
 * \f[
 * n = \sum_{i=0}^{L-1}d_i(n) b^i,
 * \f] 
 * and the \f$n\f$-th number in a van der Corput sequence is defined as:
 * \f[
 * g_n=\sum_{i=0}^{L-1}d_i(n) b^{-i-1}.
 * \f]
 *
 * Returns the n-th number in the Van der Corput sequence
 *
 * @param[in] n selects which number of the Halton sequence to return
 * @param[in] base prime number to be used as a base of the sequence
 * @returns the n-th number in the van_der_corput sequence
 *
 * @see http://en.wikipedia.org/wiki/Van_der_Corput_sequence
 */

double van_der_corput(unsigned int n, unsigned int base) {
    double retval = 0.;
    double f = 1.0 / base;
    unsigned int i = n;
    while (i > 0) {
        retval += f * (i % base);
        i = floor(i / base);
        f = f / base;
    }
    return retval;
}

} // namespace pagmo
#endif
