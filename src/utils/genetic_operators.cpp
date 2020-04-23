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

#include <random>
#include <utility>

#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

namespace pagmo
{

namespace detail
{

// Implementation of the binary crossover.
// Requires the distribution index eta_c in [1, 100], crossover probability p_cr  in[0,1] -> undefined algo behaviour
// otherwise Requires dimensions of the parent and bounds to be equal -> out of bound reads. dim_i is the integer
// dimension (integer alleles assumed at the end of the chromosome)

std::pair<vector_double, vector_double> sbx_crossover(const vector_double &parent1, const vector_double &parent2,
                                                      const std::pair<vector_double, vector_double> &bounds,
                                                      vector_double::size_type dim_i, const double p_cr,
                                                      const double eta_c, detail::random_engine_type &random_engine)
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
    if (drng(random_engine) < p_cr) {
        for (decltype(dim_c) i = 0u; i < dim_c; i++) {
            if ((drng(random_engine) < 0.5) && (std::abs(parent1[i] - parent2[i])) > 1e-14 && lb[i] != ub[i]) {
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
                if (rand01 < (1. / alpha)) {
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
                if (drng(random_engine) < .5) {
                    child1[i] = c1;
                    child2[i] = c2;
                } else {
                    child1[i] = c2;
                    child2[i] = c1;
                }
            }
        }

        // This implements two-point binary crossover and applies it to the integer part of the chromosome.
        if (dim_i > 1) {
            std::uniform_int_distribution<vector_double::size_type> ra_num(dim_c, dim - 1u);
            site1 = ra_num(random_engine);
            site2 = ra_num(random_engine);
            if (site1 > site2) {
                std::swap(site1, site2);
            }
            for (decltype(site2) j = site1; j < site2; ++j) {
                child1[j] = parent2[j];
                child2[j] = parent1[j];
            }
        }
    }
    return std::make_pair(std::move(child1), std::move(child2));
}

// Performs polynomial mutation. Requires all sizes to be consistent. Does not check if input is well formed.
// p_m is the mutation probability, eta_m the distibution index
void polynomial_mutation(vector_double &child, const std::pair<vector_double, vector_double> &bounds,
                         vector_double::size_type dim_i, const double p_m, const double eta_m,
                         detail::random_engine_type &random_engine)
{
    // Decision vector dimensions
    auto dim = child.size();
    auto dim_c = dim - dim_i;
    // Problem bounds
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    // declarations
    double rnd, delta1, delta2, mut_pow, deltaq;
    double y, yl, yu, val, xy;
    // Random distributions
    std::uniform_real_distribution<> drng(0., 1.); // to generate a number in [0, 1)
    // This implements the real polinomial mutation and applies it to the non integer part of the decision vector
    for (decltype(dim_c) j = 0u; j < dim_c; ++j) {
        if (drng(random_engine) < p_m && lb[j] != ub[j]) {
            y = child[j];
            yl = lb[j];
            yu = ub[j];
            delta1 = (y - yl) / (yu - yl);
            delta2 = (yu - y) / (yu - yl);
            rnd = drng(random_engine);
            mut_pow = 1. / (eta_m + 1.);
            if (rnd < 0.5) {
                xy = 1. - delta1;
                val = 2. * rnd + (1. - 2. * rnd) * (std::pow(xy, (eta_m + 1.)));
                deltaq = std::pow(val, mut_pow) - 1.;
            } else {
                xy = 1. - delta2;
                val = 2. * (1. - rnd) + 2. * (rnd - 0.5) * (std::pow(xy, (eta_m + 1.)));
                deltaq = 1. - (std::pow(val, mut_pow));
            }
            y = y + deltaq * (yu - yl);
            if (y < yl) y = yl;
            if (y > yu) y = yu;
            child[j] = y;
        }
    }

    // This implements the integer mutation for an individual
    for (decltype(dim) j = dim_c; j < dim; ++j) {
        if (drng(random_engine) < p_m) {
            // We need to draw a random integer in [lb, ub].
            auto mutated = uniform_integral_from_range(lb[j], ub[j], random_engine);
            child[j] = mutated;
        }
    }
}

// Multi-objective tournament selection. Requires all sizes to be consistent. Does not check if input is well formed.
vector_double::size_type mo_tournament_selection(vector_double::size_type idx1, vector_double::size_type idx2,
                                                 const std::vector<vector_double::size_type> &non_domination_rank,
                                                 const std::vector<double> &crowding_d,
                                                 detail::random_engine_type &random_engine)
{
    if (non_domination_rank[idx1] < non_domination_rank[idx2]) return idx1;
    if (non_domination_rank[idx1] > non_domination_rank[idx2]) return idx2;
    if (crowding_d[idx1] > crowding_d[idx2]) return idx1;
    if (crowding_d[idx1] < crowding_d[idx2]) return idx2;
    std::uniform_real_distribution<> drng(0., 1.); // to generate a number in [0, 1)
    return ((drng(random_engine) < 0.5) ? idx1 : idx2);
}

} // namespace detail
} // namespace pagmo
