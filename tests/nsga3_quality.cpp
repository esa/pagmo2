/* Copyright 2017-2021 PaGMO development team

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

/*  Quality regressions for the NSGA-III algorithm on the DTLZ test suite.
 *
 *  Every run is fixed-seed and therefore deterministic. The bounds below are not
 *  tuned to the value a particular seed happens to produce: each was observed over
 *  four seeds and then loosened by a wide margin, so that they detect a broken
 *  algorithm rather than an unlucky one.
 */

// NOTE: this include comes first on purpose. It is a compile-time guard that the
// public nsga3 header remains self-contained, i.e. usable as the first pagmo
// header in a translation unit.
#include <pagmo/algorithms/nsga3.hpp>

#define BOOST_TEST_MODULE nsga3_quality_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#include <pagmo/detail/nsga3_impl.hpp>
#include <pagmo/detail/reference_point.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

using namespace pagmo;

namespace
{

// The analytic Pareto front of the DTLZ problems used here
enum class front_shape {
    linear, // DTLZ1: sum(f) == 0.5
    sphere  // DTLZ2 and DTLZ4: ||f|| == 1
};

struct quality_metrics {
    double p_dist;                   // pagmo's own convergence measure, in decision space
    double nd_fraction;              // share of the population on the first non dominated front
    double coverage;                 // share of reference directions with at least one individual
    double front_error;              // mean distance to the analytic front, in objective space
    double igd;                      // inverted generational distance against the analytic front
    std::vector<vector_double> objs; // the final objectives, for cross-run comparisons
};

/*  Samples the analytic Pareto front at the structured reference directions. Deb
 *  and Jain build the IGD reference set the same way, and doing it here keeps the
 *  metric free of any data file or external dependency.
 */
std::vector<vector_double> front_reference_set(std::size_t nobj, std::size_t divisions, std::size_t divisions_inner,
                                               front_shape shape)
{
    auto directions = detail::generate_reference_directions(nobj, divisions, divisions_inner);
    std::vector<vector_double> reference;
    reference.reserve(directions.size());
    for (const auto &d : directions) {
        const auto &w = d.get_coeffs();
        vector_double z(w.size());
        if (shape == front_shape::linear) {
            // DTLZ1: the directions sum to one, the front to one half
            for (std::size_t j = 0; j < w.size(); ++j) {
                z[j] = 0.5 * w[j];
            }
        } else {
            // DTLZ2 and DTLZ4: project the direction onto the unit sphere
            double norm = 0.0;
            for (double c : w) {
                norm += c * c;
            }
            norm = std::sqrt(norm);
            for (std::size_t j = 0; j < w.size(); ++j) {
                z[j] = w[j] / norm;
            }
        }
        reference.push_back(z);
    }
    return reference;
}

/*  Inverted generational distance: the mean, over the sampled front, of the distance
 *  to the closest individual. Unlike p_distance it degrades when the population
 *  collapses onto part of the front, so it measures spread as well as convergence.
 */
double inverted_generational_distance(const std::vector<vector_double> &reference,
                                      const std::vector<vector_double> &objs)
{
    double total = 0.0;
    for (const auto &z : reference) {
        double best = std::numeric_limits<double>::max();
        for (const auto &f : objs) {
            double acc = 0.0;
            for (std::size_t j = 0; j < z.size(); ++j) {
                acc += (z[j] - f[j]) * (z[j] - f[j]);
            }
            best = std::min(best, std::sqrt(acc));
        }
        total += best;
    }
    return total / static_cast<double>(reference.size());
}

/*  Evolves the problem and asserts the invariants that must hold whatever the
 *  problem is, then returns the quality metrics for the caller to bound.
 */
quality_metrics evolve_and_check(unsigned prob_id, unsigned dim, unsigned nobj, std::size_t divisions,
                                 std::size_t divisions_inner, unsigned np, unsigned gens, unsigned seed,
                                 front_shape shape, bool random_mating = true)
{
    dtlz udp{prob_id, dim, nobj};
    problem prob{udp};
    const auto bounds = prob.get_bounds();

    population pop{prob, np, seed};
    const auto fevals0 = pop.get_problem().get_fevals();

    // The mutation rate of one over the chromosome length is the usual NSGA-III setting
    nsga3 algo{gens,          1.0,  30.,  1.0 / static_cast<double>(dim), 20., divisions, divisions_inner,
               random_mating, seed, false};
    pop = algo.evolve(pop);

    // The population survives the evolution intact
    BOOST_REQUIRE_EQUAL(pop.size(), np);
    BOOST_CHECK_EQUAL(pop.get_problem().get_fevals() - fevals0, static_cast<unsigned long long>(gens) * np);

    // Every individual is in bounds and has a finite fitness
    const auto xs = pop.get_x();
    const auto objs = pop.get_f();
    for (const auto &x : xs) {
        BOOST_REQUIRE_EQUAL(x.size(), bounds.first.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            BOOST_CHECK(x[i] >= bounds.first[i]);
            BOOST_CHECK(x[i] <= bounds.second[i]);
        }
    }
    for (const auto &f : objs) {
        BOOST_REQUIRE_EQUAL(f.size(), nobj);
        for (double value : f) {
            BOOST_CHECK(std::isfinite(value));
        }
    }

    quality_metrics m{};
    m.p_dist = udp.p_distance(pop);

    auto fronts = std::get<0>(fast_non_dominated_sorting(objs));
    m.nd_fraction = static_cast<double>(fronts[0].size()) / static_cast<double>(objs.size());

    /*  Reference direction coverage: normalize the final population through the same
     *  pipeline the algorithm uses, then associate it against the reference set. A
     *  single front makes every individual a candidate rather than a member.
     */
    std::vector<pop_size_t> selected(objs.size());
    std::iota(selected.begin(), selected.end(), pop_size_t(0));
    const std::vector<std::vector<pop_size_t>> whole(1, selected);
    auto ideal_point = detail::nsga3_compute_ideal(objs, selected, nullptr);
    auto translated = detail::nsga3_translate_objectives(objs, ideal_point);
    auto ext_points = detail::nsga3_find_extreme_points(selected, translated, ideal_point, nullptr);
    auto intercepts = detail::nsga3_find_intercepts(ext_points, translated, selected);
    auto norm_objs = detail::nsga3_normalize_objectives(translated, intercepts);
    auto rps = detail::generate_reference_directions(nobj, divisions, divisions_inner);
    detail::associate_with_reference_points(rps, norm_objs, whole);
    std::size_t covered = 0;
    for (const auto &rp : rps) {
        covered += (rp.candidate_count() > 0u) ? 1u : 0u;
    }
    m.coverage = static_cast<double>(covered) / static_cast<double>(rps.size());

    /*  Distance to the analytic front, computed from the objectives alone. This is
     *  an independent path to the same conclusion as p_distance, which works from
     *  the decision vectors.
     */
    double total = 0.0;
    for (const auto &f : objs) {
        double acc = 0.0;
        if (shape == front_shape::linear) {
            for (double c : f) {
                acc += c;
            }
            total += std::abs(acc - 0.5);
        } else {
            for (double c : f) {
                acc += c * c;
            }
            total += std::abs(std::sqrt(acc) - 1.0);
        }
    }
    m.front_error = total / static_cast<double>(objs.size());

    m.igd = inverted_generational_distance(front_reference_set(nobj, divisions, divisions_inner, shape), objs);
    m.objs = objs;

    return m;
}

} // namespace

BOOST_AUTO_TEST_CASE(nsga3_quality_dtlz1_3obj)
{
    /*  12 divisions over 3 objectives is Deb and Jain's Table I setting: 91
     *  directions for 92 individuals.
     *
     *  DTLZ1 is multimodal, and it is the case where the canonical random mating of
     *  Section IV-F is most exposed: without any selection pressure at mating,
     *  escaping the local fronts is slow. Observed across four seeds at 200
     *  generations: p_distance 0.03 to 1.08, coverage 0.49 to 0.93, IGD 0.019 to
     *  0.216. The bounds sit well outside those ranges. The same runs with the
     *  tournament mating stay below a p_distance of 0.08, which is why the option
     *  exists.
     */
    auto m = evolve_and_check(1u, 7u, 3u, 12u, 0u, 92u, 200u, 32u, front_shape::linear);
    BOOST_CHECK(m.p_dist < 2.0);
    BOOST_CHECK(m.nd_fraction > 0.7);
    BOOST_CHECK(m.coverage > 0.3);
    BOOST_CHECK(m.front_error < 1.0);
    BOOST_CHECK(m.igd < 0.5);
}

BOOST_AUTO_TEST_CASE(nsga3_quality_dtlz2_3obj)
{
    /*  Table I again: 12 divisions, 91 directions, 92 individuals. Observed across
     *  four seeds: p_distance 0.0030 to 0.0036, full coverage, IGD below 0.009.
     */
    auto m = evolve_and_check(2u, 10u, 3u, 12u, 0u, 92u, 100u, 32u, front_shape::sphere);
    BOOST_CHECK(m.p_dist < 0.05);
    BOOST_CHECK(m.nd_fraction > 0.9);
    BOOST_CHECK(m.coverage > 0.8);
    BOOST_CHECK(m.front_error < 0.05);
    BOOST_CHECK(m.igd < 0.05);
}

BOOST_AUTO_TEST_CASE(nsga3_quality_dtlz2_5obj)
{
    /*  Table I: 6 divisions over 5 objectives gives 210 directions for 212
     *  individuals. Observed across four seeds: p_distance 0.012 to 0.014, full
     *  coverage, IGD below 0.026.
     */
    auto m = evolve_and_check(2u, 10u, 5u, 6u, 0u, 212u, 100u, 32u, front_shape::sphere);
    BOOST_CHECK(m.p_dist < 0.1);
    BOOST_CHECK(m.nd_fraction > 0.9);
    BOOST_CHECK(m.coverage > 0.8);
    BOOST_CHECK(m.front_error < 0.1);
    BOOST_CHECK(m.igd < 0.15);
}

BOOST_AUTO_TEST_CASE(nsga3_quality_dtlz2_8obj_two_layer)
{
    /*  The eight-objective row of Table I, which the two layer construction makes
     *  reachable: 3 divisions on the boundary layer and 2 on the inside layer give
     *  120 + 36 = 156 reference directions, and the population is exactly 156 as
     *  well. This is also the case which requires the population size rule to admit
     *  equality between the two.
     *
     *  Observed across four seeds: p_distance 0.010 to 0.020, full coverage, IGD
     *  0.056 to 0.075.
     */
    BOOST_REQUIRE_EQUAL(detail::generate_reference_directions(8u, 3u, 2u).size(), 156u);
    auto m = evolve_and_check(2u, 10u, 8u, 3u, 2u, 156u, 100u, 32u, front_shape::sphere);
    BOOST_CHECK(m.p_dist < 0.1);
    BOOST_CHECK(m.nd_fraction > 0.9);
    BOOST_CHECK(m.coverage > 0.8);
    BOOST_CHECK(m.front_error < 0.1);
    BOOST_CHECK(m.igd < 0.25);
}

BOOST_AUTO_TEST_CASE(nsga3_quality_dtlz4)
{
    /*  DTLZ4 biases solutions towards a subset of the front, and the population is
     *  known to be able to collapse onto a few directions. With the canonical random
     *  mating all four seeds kept full coverage and an IGD below 0.017, but the
     *  collapse is the problem behaving as designed rather than a defect, so
     *  coverage is only required to be non-zero and the IGD bound is left loose.
     *  p_distance stayed below 0.003 throughout.
     */
    auto m = evolve_and_check(4u, 10u, 3u, 12u, 0u, 92u, 100u, 32u, front_shape::sphere);
    BOOST_CHECK(m.p_dist < 0.1);
    BOOST_CHECK(m.nd_fraction > 0.9);
    BOOST_CHECK(m.front_error < 0.1);
    BOOST_CHECK(m.coverage > 0.0);
    BOOST_CHECK(m.igd < 1.5);
}

BOOST_AUTO_TEST_CASE(nsga3_quality_tournament_mating)
{
    /*  The non-default mating scheme must converge too. Observed across four seeds
     *  at 100 generations: p_distance 0.0024 to 0.0037, coverage at least 0.99, IGD
     *  below 0.010.
     */
    auto m = evolve_and_check(2u, 10u, 3u, 12u, 0u, 92u, 60u, 32u, front_shape::sphere, false);
    BOOST_CHECK(m.p_dist < 0.05);
    BOOST_CHECK(m.nd_fraction > 0.9);
    BOOST_CHECK(m.coverage > 0.8);
    BOOST_CHECK(m.igd < 0.05);
}

BOOST_AUTO_TEST_CASE(nsga3_quality_seed_sensitivity)
{
    /*  Different seeds must explore differently and still arrive somewhere good.
     *  Asserting only that the runs differ would pass for an algorithm that had
     *  stopped working, so each run is held to the same bounds independently.
     */
    const std::vector<unsigned> seeds{32u, 7u, 101u};
    std::vector<std::vector<vector_double>> populations;

    for (unsigned seed : seeds) {
        auto m = evolve_and_check(2u, 10u, 3u, 12u, 0u, 92u, 100u, seed, front_shape::sphere);
        BOOST_CHECK(m.p_dist < 0.05);
        BOOST_CHECK(m.nd_fraction > 0.9);
        BOOST_CHECK(m.coverage > 0.8);
        BOOST_CHECK(m.igd < 0.05);
        populations.push_back(m.objs);
    }

    // Every pair of runs really did explore a different part of the space
    for (std::size_t i = 0; i < populations.size(); ++i) {
        for (std::size_t j = i + 1; j < populations.size(); ++j) {
            BOOST_CHECK(populations[i] != populations[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga3_quality_reproducible)
{
    /*  The quality suite is itself deterministic: the same seed must produce the
     *  same population, so a threshold that holds once holds every time.
     */
    dtlz udp{2u, 10u, 3u};
    population pop_a{udp, 92u, 32u};
    population pop_b{udp, 92u, 32u};
    nsga3 algo_a{40u, 1.0, 30., 0.1, 20., 12u, 0u, true, 32u, false};
    nsga3 algo_b{40u, 1.0, 30., 0.1, 20., 12u, 0u, true, 32u, false};

    pop_a = algo_a.evolve(pop_a);
    pop_b = algo_b.evolve(pop_b);

    BOOST_CHECK(pop_a.get_x() == pop_b.get_x());
    BOOST_CHECK(pop_a.get_f() == pop_b.get_f());
}
