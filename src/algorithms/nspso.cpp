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

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/serialization/optional.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nspso.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/multi_objective.hpp>

namespace pagmo
{

nspso::nspso(unsigned gen, double omega, double c1, double c2, double chi, double v_coeff,
             unsigned leader_selection_range, std::string diversity_mechanism, bool memory, unsigned seed)
    : m_gen(gen), m_omega(omega), m_c1(c1), m_c2(c2), m_chi(chi), m_v_coeff(v_coeff),
      m_leader_selection_range(leader_selection_range), m_diversity_mechanism(diversity_mechanism), m_memory(memory),
      m_velocity(), m_e(seed), m_seed(seed), m_verbosity(0u)
{
    if (omega < 0. || omega > 1.) {
        pagmo_throw(std::invalid_argument,
                    "The particles' inertia weight must be in the [0,1] range, while a value of "
                        + std::to_string(m_omega) + " was detected");
    }
    if (c1 <= 0 || c2 <= 0 || chi <= 0) {
        pagmo_throw(std::invalid_argument, "first and second magnitude of the force "
                                           "coefficients and velocity scaling factor should be greater than 0");
    }
    if (v_coeff <= 0 || v_coeff > 1) {
        pagmo_throw(std::invalid_argument,
                    "velocity scaling factor should be in ]0,1] range, while a value of" + std::to_string(v_coeff)
                        + " was detected");
    }
    if (leader_selection_range > 100) {
        pagmo_throw(std::invalid_argument,
                    "leader selection range coefficient should be in the ]0,100] range, while a value of"
                        + std::to_string(leader_selection_range) + " was detected");
    }
    if (diversity_mechanism != "crowding distance" && diversity_mechanism != "niche count"
        && diversity_mechanism != "max min") {
        pagmo_throw(std::invalid_argument, "Non existing diversity mechanism method.");
    }
}

// Algorithm evolve method
population nspso::evolve(population pop) const
{
    // We store some useful variables
    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                          // allowed
    auto n_x = prob.get_nx();             // Sum of continuous+integer dimensions
    auto bounds = prob.get_bounds();
    auto &lb = bounds.first;
    auto &ub = bounds.second;
    auto swarm_size = pop.size();
    unsigned count_verb = 1u; // regulates the screen output

    // PREAMBLE-------------------------------------------------------------------------------------------------
    // We start by checking that the problem is suitable for this
    // particular algorithm.
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument,
                    "The problem appears to be stochastic " + get_name() + " cannot deal with it");
    }
    if (prob.get_nc() != 0u) {
        pagmo_throw(std::invalid_argument,
                    "Non linear constraints detected in " + prob.get_name() + " instance. " + get_name()
                        + " cannot deal with them.");
    }
    if (prob.get_nf() < 2u) {
        pagmo_throw(std::invalid_argument,
                    "This is a multi-objective algortihm, while number of objectives detected in " + prob.get_name()
                        + " is " + std::to_string(prob.get_nf()));
    }
    if (pop.size() <= 1u) {
        pagmo_throw(std::invalid_argument,
                    get_name() + " can only work with population sizes >=2, whereas " + std::to_string(pop.size())
                        + " were detected.");
    }
    // Get out if there is nothing to do.
    if (m_gen == 0u) {
        return pop;
    }
    // No throws, all valid: we clear the logs
    m_log.clear();

    vector_double dummy_vel(n_x, 0.);           // used for initialisation purposes
    vector_double dummy_fit(prob.get_nf(), 0.); // used for initialisation purposes
    vector_double minv(n_x), maxv(n_x);         // Maximum and minimum velocity allowed
    std::uniform_real_distribution<double> drng_real(0.0, 1.0);
    std::vector<vector_double::size_type> sort_list_2(swarm_size);
    std::vector<vector_double::size_type> sort_list_3(2 * swarm_size);
    if (m_best_fit.size() != pop.get_f().size() && m_memory == true) {
        m_best_fit = pop.get_f();
        m_best_dvs = pop.get_x();
    } else if (m_memory == false) {
        m_best_fit = pop.get_f();
        m_best_dvs = pop.get_x();
    }
    std::vector<vector_double> next_pop_fit(2 * swarm_size, dummy_fit);
    std::vector<vector_double> next_pop_dvs(2 * swarm_size);

    double vwidth; // Temporary variable
    // Initialise the minimum and maximum velocity
    for (decltype(n_x) i = 0u; i < n_x; ++i) {
        vwidth = (ub[i] - lb[i]) * m_v_coeff;
        minv[i] = -1. * vwidth;
        maxv[i] = vwidth;
    }
    // Initialize the particle velocities if necessary
    if ((m_velocity.size() != swarm_size) || (!m_memory)) {
        m_velocity = std::vector<vector_double>(swarm_size, dummy_vel);
        for (decltype(swarm_size) i = 0u; i < swarm_size; ++i) {
            for (decltype(n_x) j = 0u; j < n_x; ++j) {
                m_velocity[i][j] = uniform_real_from_range(minv[j], maxv[j], m_e);
            }
        }
    }

    // Main NSPSO loop
    for (decltype(m_gen) gen = 1u; gen <= m_gen; gen++) {
        std::vector<vector_double::size_type> best_non_dom_indices;
        auto fit = pop.get_f();
        auto dvs = pop.get_x();
        // This returns a std::tuple containing: -the non dominated fronts, -the domination list, -the domination
        // count, -the non domination rank
        auto fnds_res = fast_non_dominated_sorting(fit);
        // 0 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
        if (m_verbosity > 0u) {
            // Every m_verbosity generations print a log line
            if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                // We compute the ideal point
                auto ideal_point_verb = ideal(m_best_fit);
                // Every 50 lines print the column names
                if (count_verb % 50u == 1u) {
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:");
                    for (decltype(ideal_point_verb.size()) i = 0u; i < ideal_point_verb.size(); ++i) {
                        if (i >= 5u) {
                            print(std::setw(15), "... :");
                            break;
                        }
                        print(std::setw(15), "ideal" + std::to_string(i + 1u) + ":");
                    }
                    print('\n');
                }
                print(std::setw(7), gen, std::setw(15), prob.get_fevals());
                for (decltype(ideal_point_verb.size()) i = 0u; i < ideal_point_verb.size(); ++i) {
                    if (i >= 5u) {
                        break;
                    }
                    print(std::setw(15), ideal_point_verb[i]);
                }
                print('\n');
                ++count_verb;
                // Logs
                m_log.emplace_back(gen, prob.get_fevals(), ideal_point_verb);
            }
        }

        // 1 - Calculate non-dominated population
        if (m_diversity_mechanism == "crowding distance") {
            auto ndf = std::get<0>(fnds_res);
            auto best_non_dom_indices_tmp = sort_population_mo(fit);
            std::vector<vector_double::size_type> dummy(ndf[0].size());
            for (decltype(dummy.size()) i = 0u; i < dummy.size(); ++i) {
                dummy[i] = best_non_dom_indices_tmp[i];
            }
            if (ndf[0].size() > 1) {
                best_non_dom_indices = std::vector<vector_double::size_type>(
                    dummy.begin(), dummy.begin() + static_cast<vector_double::difference_type>(ndf[0].size()));
            } else { // ensure the non-dom population has at least 2 individuals (to avoid convergence to a point)
                best_non_dom_indices = std::vector<vector_double::size_type>(
                    best_non_dom_indices_tmp.begin(),
                    best_non_dom_indices_tmp.begin() + static_cast<vector_double::difference_type>(2));
            }

        } else if (m_diversity_mechanism == "niche count") {
            auto ndf = std::get<0>(fnds_res);
            auto best_ndi_tmp = sort_population_mo(fit);
            std::vector<vector_double> non_dom_chromosomes(ndf[0].size());

            for (decltype(ndf[0].size()) i = 0u; i < ndf[0].size(); ++i) {
                non_dom_chromosomes[i] = dvs[ndf[0][i]];
            }
            vector_double nadir_point = nadir(fit);
            vector_double ideal_point = ideal(fit);

            // Fonseca-Fleming setting for delta
            double delta = 1.0;
            if (prob.get_nobj() == 2) {
                vector_double::size_type dummy_delta = non_dom_chromosomes.size();
                if (non_dom_chromosomes.size() == 1) {
                    dummy_delta = 2;
                }
                delta = ((nadir_point[0] - ideal_point[0]) + (nadir_point[1] - ideal_point[1]))
                        / (static_cast<double>(dummy_delta) - 1);
            } else if (prob.get_nobj() == 3) {
                double d1 = nadir_point[0] - ideal_point[0];
                double d2 = nadir_point[1] - ideal_point[1];
                double d3 = nadir_point[2] - ideal_point[2];
                double ndc_size = static_cast<double>(non_dom_chromosomes.size());
                if (ndc_size < 2.0) {
                    ndc_size = 2.0;
                }
                delta = std::sqrt(4 * d2 * d1 * ndc_size + 4 * d3 * d1 * ndc_size + 4 * d2 * d3 * ndc_size
                                  + std::pow(d1, 2) + std::pow(d2, 2) + std::pow(d3, 2) - 2 * d2 * d1 - 2 * d3 * d1
                                  - 2 * d2 * d3 + d1 + d2 + d3)
                        / (2 * (ndc_size - 1));
            } else { // for higher dimension we just divide equally the entire volume containing the pareto front
                for (decltype(nadir_point.size()) i = 0; i < nadir_point.size(); ++i) {
                    delta *= nadir_point[i] - ideal_point[i];
                }
                delta = pow(delta, 1.0 / static_cast<double>(nadir_point.size()))
                        / static_cast<double>(non_dom_chromosomes.size());
            }

            std::vector<vector_double::size_type> count(non_dom_chromosomes.size(), 0);
            std::vector<vector_double::size_type> sort_list(non_dom_chromosomes.size());
            compute_niche_count(count, non_dom_chromosomes, delta);
            std::iota(std::begin(sort_list), std::end(sort_list), vector_double::size_type(0));
            std::sort(
                sort_list.begin(), sort_list.end(), [&count](decltype(count.size()) idx1, decltype(count.size()) idx2) {
                    return detail::less_than_f(static_cast<double>(count[idx1]), static_cast<double>(count[idx2]));
                });

            if (ndf[0].size() > 1) {
                for (decltype(sort_list.size()) i = 0; i < sort_list.size(); ++i) {
                    best_non_dom_indices.push_back(ndf[0][sort_list[i]]);
                }
            } else { // ensure the non-dom population has at least 2 individuals (to avoid convergence to a point)
                best_non_dom_indices.push_back(ndf[0][0]);
                best_non_dom_indices.push_back(ndf[1][0]);
            }
        } else { // m_diversity_method == max min
            vector_double maxmin(swarm_size, 0);
            compute_maxmin(maxmin, fit);

            std::iota(std::begin(sort_list_2), std::end(sort_list_2), vector_double::size_type(0));
            std::sort(sort_list_2.begin(), sort_list_2.end(),
                      [&maxmin](decltype(maxmin.size()) idx1, decltype(maxmin.size()) idx2) {
                          return detail::less_than_f(maxmin[idx1], maxmin[idx2]);
                      });
            best_non_dom_indices = sort_list_2;
            vector_double::size_type i;
            for (i = 1u; i < best_non_dom_indices.size() && maxmin[best_non_dom_indices[i]] < 0; ++i)
                ;
            if (i < 2) {
                i = 2; // ensure the non-dom population has at least 2 individuals (to avoid convergence to a point)
            }

            best_non_dom_indices = std::vector<vector_double::size_type>(
                best_non_dom_indices.begin(),
                best_non_dom_indices.begin()
                    + static_cast<vector_double::difference_type>(i)); // keep just the non dominated
        }

        // 2 - Move the points
        for (decltype(swarm_size) idx = 0; idx < swarm_size; ++idx) {
            // Calculate the leader
            int ext = static_cast<int>(ceil(static_cast<double>(best_non_dom_indices.size())
                                            * static_cast<double>(m_leader_selection_range) / 100.0)
                                       - 1);

            if (ext < 1) {
                ext = 1;
            }

            vector_double::size_type leader_idx;
            do {
                std::uniform_int_distribution<int> drng(0, ext);
                leader_idx
                    = static_cast<vector_double::size_type>(drng(m_e)); // to generate an integer number in [0, ext]
            } while (best_non_dom_indices[leader_idx] == idx);
            vector_double leader = m_best_dvs[best_non_dom_indices[leader_idx]];

            // Calculate some random factors
            double r1 = drng_real(m_e);
            double r2 = drng_real(m_e);

            // Calculate new velocity and new position for each particle
            vector_double new_dvs(n_x);
            vector_double new_vel(n_x);
            for (decltype(n_x) i = 0; i < n_x; ++i) {
                double v = m_omega * m_velocity[idx][i] + m_c1 * r1 * (m_best_dvs[idx][i] - dvs[idx][i])
                           + m_c2 * r2 * (leader[i] - dvs[idx][i]);
                if (v > maxv[i]) {
                    v = maxv[i];
                } else if (v < minv[i]) {
                    v = minv[i];
                }
                double x = dvs[idx][i] + m_chi * v;
                if (x > ub[i]) {
                    x = ub[i];
                    v = 0.0;

                } else if (x < lb[i]) {
                    x = lb[i];
                    v = 0.0;
                }
                new_vel[i] = v;
                new_dvs[i] = x;
            }
            // Add the moved particle to the population
            dvs[idx] = new_dvs;
            m_velocity[idx] = new_vel;
            next_pop_dvs[idx] = new_dvs;
            if (!m_bfe) {
                // bfe not available
                fit[idx] = prob.fitness(new_dvs);
                next_pop_fit[idx] = fit[idx];
            }
        }
        if (m_bfe) {
            // bfe is available:
            vector_double decision_vectors(swarm_size * dvs[0].size());
            vector_double::size_type pos = 0u;
            for (decltype(swarm_size) i = 0u; i < swarm_size; ++i) {
                for (decltype(dvs[0].size()) ii = 0u; ii < dvs[0].size(); ++ii) {
                    decision_vectors[pos] = dvs[i][ii];
                    ++pos;
                }
            }
            // run bfe.
            auto fitnesses = (*m_bfe)(prob, decision_vectors);
            vector_double::size_type pos_fit = 0u;
            for (decltype(swarm_size) i = 0; i < swarm_size; ++i) {
                for (decltype(fit[0].size()) ii_f = 0u; ii_f < fit[0].size(); ++ii_f) {
                    fit[i][ii_f] = fitnesses[pos_fit];
                    next_pop_fit[i][ii_f] = fitnesses[pos_fit];
                    ++pos_fit;
                }
            }
        }
        // 3 - Select the best swarm_size individuals in the new population (of size 2*swarm_size) according to pareto
        // dominance
        for (decltype(swarm_size) i = swarm_size; i < 2 * swarm_size; ++i) {
            next_pop_fit[i] = m_best_fit[i - swarm_size];
            next_pop_dvs[i] = m_best_dvs[i - swarm_size];
        }
        std::vector<vector_double::size_type> best_next_pop_indices(swarm_size, 0);
        if (m_diversity_mechanism != "max min") {
            auto fnds_res_next = fast_non_dominated_sorting(next_pop_fit);
            auto best_next_pop_indices_tmp = sort_population_mo(next_pop_fit);
            for (decltype(swarm_size) i = 0u; i < swarm_size; ++i) {
                best_next_pop_indices[i] = best_next_pop_indices_tmp[i];
            }
        } else { // "max min" diversity mechanism
            vector_double maxmin(2 * swarm_size, 0);
            compute_maxmin(maxmin, next_pop_fit);
            // I extract the index list of maxmin sorted:
            std::iota(std::begin(sort_list_3), std::end(sort_list_3), vector_double::size_type(0));
            std::sort(sort_list_3.begin(), sort_list_3.end(),
                      [&maxmin](decltype(maxmin.size()) idx1, decltype(maxmin.size()) idx2) {
                          return detail::less_than_f(maxmin[idx1], maxmin[idx2]);
                      });
            best_next_pop_indices = std::vector<vector_double::size_type>(
                sort_list_3.begin(), sort_list_3.begin() + static_cast<vector_double::difference_type>(swarm_size));
        }

        // The next_pop_list for the next generation will contain the best swarm_size individuals out of 2*swarm_size
        // according to pareto dominance
        for (decltype(swarm_size) i = 0u; i < swarm_size; ++i) {
            m_best_dvs[i] = next_pop_dvs[best_next_pop_indices[i]];
            m_best_fit[i] = next_pop_fit[best_next_pop_indices[i]];
        }
        // 4 - I now copy insert the new population
        for (decltype(swarm_size) i = 0; i < swarm_size; ++i) {
            pop.set_xf(i, dvs[i], fit[i]);
        }

    } // end of main NSPSO loop
    return pop;
}

// Sets the seed
void nspso::set_seed(unsigned seed)
{
    m_e.seed(seed);
    m_seed = seed;
}

// Sets the batch function evaluation scheme
void nspso::set_bfe(const bfe &b)
{
    m_bfe = b;
}

// Extra info
std::string nspso::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tGenerations: ", m_gen);
    stream(ss, "\n\tInertia weight: ", m_omega);
    stream(ss, "\n\tFirst magnitude of the force coefficients: ", m_c1);
    stream(ss, "\n\tSecond magnitude of the force coefficients: ", m_c2);
    stream(ss, "\n\tVelocity scaling factor: ", m_chi);
    stream(ss, "\n\tVelocity coefficient: ", m_v_coeff);
    stream(ss, "\n\tLeader selection range: ", m_leader_selection_range);
    stream(ss, "\n\tDiversity mechanism: ", m_diversity_mechanism);
    stream(ss, "\n\tSeed: ", m_seed);
    stream(ss, "\n\tVerbosity: ", m_verbosity);
    return ss.str();
}

// Object serialization
template <typename Archive>
void nspso::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_gen, m_omega, m_c1, m_c2, m_chi, m_v_coeff, m_leader_selection_range, m_diversity_mechanism,
                    m_e, m_seed, m_verbosity, m_log, m_bfe);
}

double nspso::minfit(vector_double::size_type(i), vector_double::size_type(j),
                     const std::vector<vector_double> &fit) const

{

    double min = fit[i][0] - fit[j][0];

    for (decltype(fit[0].size()) f = 0; f < fit[0].size(); ++f) {

        double tmp = fit[i][f] - fit[j][f];

        if (tmp < min) {

            min = tmp;
        }
    }

    return min;
}

void nspso::compute_maxmin(vector_double &maxmin, const std::vector<vector_double> &fit) const

{
    for (decltype(fit.size()) i = 0; i < fit.size(); ++i) {

        maxmin[i] = minfit(i, (i + 1) % fit.size(), fit);

        for (decltype(fit.size()) j = 0; j < fit.size(); ++j) {

            if (i != j) {

                double tmp = minfit(i, j, fit);

                if (tmp > maxmin[i]) {

                    maxmin[i] = tmp;
                }
            }
        }
    }
}

double nspso::euclidian_distance(const vector_double &x, const vector_double &y) const

{
    double sum = 0.0;

    for (decltype(x.size()) i = 0; i < x.size(); ++i) {

        sum += pow(x[i] - y[i], 2);
    }

    return sqrt(sum);
}

void nspso::compute_niche_count(std::vector<vector_double::size_type> &count,
                                const std::vector<vector_double> &chromosomes, double delta) const

{

    std::fill(count.begin(), count.end(), 0);

    for (decltype(chromosomes.size()) i = 0; i < chromosomes.size(); ++i) {

        for (decltype(chromosomes.size()) j = 0; j < chromosomes.size(); ++j) {

            if (euclidian_distance(chromosomes[i], chromosomes[j]) < delta) {

                ++count[i];
            }
        }
    }
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::nspso)
