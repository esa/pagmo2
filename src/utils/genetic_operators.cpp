/* Copyright 2017-2020 PaGMO development team

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

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

#include <utility>
#include <random>

#include <pagmo/types.hpp>
#include <pagmo/rng.hpp>


namespace pagmo
{

namespace detail
{

// Implementation of the binary crossover. 
// Requires: distribution index eta_c [1, 100], crossover cr [0,1] -> UB otherwise
std::pair<vector_double, vector_double> sbx_crossover(const vector_double &parent1, const vector_double &parent2,
                                                      const std::pair<vector_double, vector_double> &bounds,
                                                      vector_double::size_type dim_i, const double p_cr, const double eta_c, detail::random_engine_type& random_engine) 
{
    // Decision vector dimensions
    auto dim = parent1.size();
    auto dim_c = dim - dim_i;
    // Problem bounds
    const vector_double &lb = bounds.first;
    const vector_double &ub = bounds.second;
    // declarations
    double y1, y2, yl, yu, rand01, beta, alpha, betaq, c1, c2;
    vector_double::size_type site1, site2;
    // Initialize the child decision vectors
    vector_double child1 = parent1;
    vector_double child2 = parent2;
    // Random distributions
    std::uniform_real_distribution<> drng(0., 1.); // to generate a number in [0, 1)

    // This implements a Simulated Binary Crossover SBX and applies it to the non integer part of the decision
    // vector
    if (drng(random_engine) <= p_cr) {
        for (decltype(dim_c) i = 0u; i < dim_c; i++) {
            if ((drng(random_engine) <= 0.5) && (std::abs(parent1[i] - parent2[i])) > 1e-14 && lb[i] != ub[i]) {
                if (parent1[i] < parent2[i]) {
                    y1 = parent1[i];
                    y2 = parent2[i];
                } else {
                    y1 = parent2[i];
                    y2 = parent1[i];
                }
                yl = lb[i];
                yu = ub[i];
                rand01 = drng(random_engine);
                beta = 1. + (2. * (y1 - yl) / (y2 - y1));
                alpha = 2. - std::pow(beta, -(eta_c + 1.));
                if (rand01 <= (1. / alpha)) {
                    betaq = std::pow((rand01 * alpha), (1. / (eta_c + 1.)));
                } else {
                    betaq = std::pow((1. / (2. - rand01 * alpha)), (1. / (eta_c + 1.)));
                }
                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
                beta = 1. + (2. * (yu - y2) / (y2 - y1));
                alpha = 2. - std::pow(beta, -(eta_c + 1.));
                if (rand01 <= (1. / alpha)) {
                    betaq = std::pow((rand01 * alpha), (1. / (eta_c + 1.)));
                } else {
                    betaq = std::pow((1. / (2. - rand01 * alpha)), (1. / (eta_c + 1.)));
                }
                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));
                if (c1 < lb[i]) c1 = lb[i];
                if (c2 < lb[i]) c2 = lb[i];
                if (c1 > ub[i]) c1 = ub[i];
                if (c2 > ub[i]) c2 = ub[i];
                if (drng(random_engine) <= .5) {
                    child1[i] = c1;
                    child2[i] = c2;
                } else {
                    child1[i] = c2;
                    child2[i] = c1;
                }
            }
        }
    }
    // This implements two-point binary crossover and applies it to the integer part of the chromosome
    for (decltype(dim_c) i = dim_c; i < dim; ++i) {
        // in this loop we are sure dim_i is at least 1
        std::uniform_int_distribution<vector_double::size_type> ra_num(0, dim_i - 1u);
        if (drng(random_engine) <= p_cr) {
            site1 = ra_num(random_engine);
            site2 = ra_num(random_engine);
            if (site1 > site2) {
                std::swap(site1, site2);
            }
            for (decltype(site1) j = 0u; j < site1; ++j) {
                child1[j] = parent1[j];
                child2[j] = parent2[j];
            }
            for (decltype(site2) j = site1; j < site2; ++j) {
                child1[j] = parent2[j];
                child2[j] = parent1[j];
            }
            for (decltype(dim_i) j = site2; j < dim_i; ++j) {
                child1[j] = parent1[j];
                child2[j] = parent2[j];
            }
        } else {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }
    }
    return std::make_pair(std::move(child1), std::move(child2));
}

} // namespace detail
} // namespace pagmo
