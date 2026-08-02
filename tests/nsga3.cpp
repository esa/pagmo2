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

// NOTE: this include comes first on purpose. It is a compile-time guard that nsga3
// is reachable through the umbrella header, i.e. that it has been added to it.
#include <pagmo/pagmo.hpp>

#define BOOST_TEST_MODULE nsga3_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/lexical_cast.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga3.hpp>
#include <pagmo/batch_evaluators/default_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/nsga3_impl.hpp>
#include <pagmo/detail/reference_point.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

using namespace pagmo;

/*  A batch fitness evaluator which evaluates strictly in order, one individual at a
 *  time. It therefore produces exactly the values of the scalar path, which is what
 *  makes it usable to compare the two paths of evolve().
 *
 *  The evaluations go through a copy of the problem, as pagmo::thread_bfe does: the
 *  function evaluation counter of the problem which was passed in is incremented
 *  once per decision vector by pagmo::bfe itself, so counting them here as well
 *  would count each of them twice.
 */
struct serial_bfe {
    vector_double operator()(const problem &p, const vector_double &dvs) const
    {
        problem prob_copy(p);
        const auto nx = prob_copy.get_nx();
        const auto n = dvs.size() / nx;
        vector_double retval;
        retval.reserve(n * prob_copy.get_nf());
        for (decltype(dvs.size()) i = 0u; i < n; ++i) {
            const vector_double dv(dvs.begin() + static_cast<vector_double::difference_type>(i * nx),
                                   dvs.begin() + static_cast<vector_double::difference_type>((i + 1u) * nx));
            const auto f = prob_copy.fitness(dv);
            retval.insert(retval.end(), f.begin(), f.end());
        }
        return retval;
    }
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

PAGMO_S11N_BFE_EXPORT(serial_bfe)

namespace
{

// Sum of the coefficients of a reference direction
double direction_sum(const detail::reference_point &rp)
{
    double sum = 0.0;
    for (double c : rp.get_coeffs()) {
        sum += c;
    }
    return sum;
}

} // namespace

BOOST_AUTO_TEST_CASE(nsga3_algorithm_construction)
{
    BOOST_CHECK_NO_THROW(nsga3{});
    nsga3 user_algo{1u, 1.00, 30.0, 0.10, 20.0, 12u, 0u, true, 32u, false};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 32u);
    BOOST_CHECK(user_algo.get_extra_info().find("Seed: 32") != std::string::npos);

    // The algorithm is reachable, and usable, through the umbrella header
    algorithm algo{nsga3{}};
    BOOST_CHECK(algo.extract<nsga3>() != nullptr);

    // Verify throw on invalid arguments
    // Invalid cr
    BOOST_CHECK_THROW((nsga3{1u, 2.00, 30.0, 0.10, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, -1.00, 30.0, 0.10, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    // Invalid mut
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 1.10, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, -0.10, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    // Invalid eta_c
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 0.5, 0.10, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 100.1, 0.10, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    // Invalid eta_mut
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 0.10, 100.1, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 0.10, -0.1, 12u, 0u, true, 32u, false}), std::invalid_argument);

    /*  Every comparison involving a NaN is false, so a plain range check would let
     *  one through. Non-finite arguments are rejected explicitly.
     */
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    BOOST_CHECK_THROW((nsga3{1u, nan, 30.0, 0.10, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, inf, 30.0, 0.10, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, -inf, 30.0, 0.10, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, nan, 0.10, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, inf, 0.10, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, nan, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, inf, 20.0, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 0.10, nan, 12u, 0u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 0.10, inf, 12u, 0u, true, 32u, false}), std::invalid_argument);

    // A layer with no divisions is not a layer
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 0.10, 20.0, 0u, 0u, true, 32u, false}), std::invalid_argument);
    // The inner layer must be no finer than the outer one
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 0.10, 20.0, 2u, 3u, true, 32u, false}), std::invalid_argument);
    BOOST_CHECK_NO_THROW((nsga3{1u, 1.00, 30.0, 0.10, 20.0, 3u, 3u, true, 32u, false}));
    BOOST_CHECK_NO_THROW((nsga3{1u, 1.00, 30.0, 0.10, 20.0, 3u, 2u, true, 32u, false}));
}

BOOST_AUTO_TEST_CASE(nsga3_name_and_extra_info)
{
    nsga3 algo{7u, 0.9, 20.0, 0.05, 15.0, 3u, 2u, false, 32u, true};

    // A descriptive pagmo style name, without a trailing colon
    const auto name = algo.get_name();
    BOOST_CHECK(name.find("NSGA-III") != std::string::npos);
    BOOST_CHECK(!name.empty());
    BOOST_CHECK(name.back() != ':');

    // Every constructor argument is reported
    const auto info = algo.get_extra_info();
    BOOST_CHECK(info.find("Generations: 7") != std::string::npos);
    BOOST_CHECK(info.find("Reference direction divisions: 3") != std::string::npos);
    BOOST_CHECK(info.find("Reference direction inner divisions: 2") != std::string::npos);
    BOOST_CHECK(info.find("Random mating: false") != std::string::npos);
    BOOST_CHECK(info.find("Inter-generational memory: true") != std::string::npos);
    BOOST_CHECK(info.find("Seed: 32") != std::string::npos);

    algo.set_verbosity(4u);
    BOOST_CHECK_EQUAL(algo.get_verbosity(), 4u);
    BOOST_CHECK(algo.get_extra_info().find("Verbosity: 4") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(nsga3_evolve_population)
{
    dtlz udp{1u, 10u, 3u};
    problem prob{udp};
    const auto bounds = prob.get_bounds();

    population pop1{prob, 92u, 23u /*seed*/};
    const auto fevals0 = pop1.get_problem().get_fevals();

    nsga3 user_algo1{10u, 1.0, 30., 0.10, 20., 12u, 0u, true, 32u, false};
    BOOST_CHECK(user_algo1.get_seed() == 32u);
    user_algo1.set_verbosity(10u);
    pop1 = user_algo1.evolve(pop1);

    // The population size is preserved across the evolution
    BOOST_CHECK_EQUAL(pop1.size(), 92u);
    // Every individual is a usable, in-bounds solution with a finite fitness
    for (const auto &x : pop1.get_x()) {
        BOOST_REQUIRE_EQUAL(x.size(), bounds.first.size());
        for (std::size_t i = 0u; i < x.size(); ++i) {
            BOOST_CHECK(x[i] >= bounds.first[i]);
            BOOST_CHECK(x[i] <= bounds.second[i]);
        }
    }
    for (const auto &f : pop1.get_f()) {
        BOOST_REQUIRE_EQUAL(f.size(), prob.get_nobj());
        for (double value : f) {
            BOOST_CHECK(std::isfinite(value));
        }
    }
    // Each generation evaluates a full offspring population
    BOOST_CHECK_EQUAL(pop1.get_problem().get_fevals() - fevals0, 10u * 92u);
}

BOOST_AUTO_TEST_CASE(nsga3_evolve_rejects_unsuitable_problems)
{
    // Single objective
    {
        population pop{problem{rosenbrock{10u}}, 32u, 23u};
        BOOST_CHECK_THROW(nsga3{}.evolve(pop), std::invalid_argument);
    }
    // Constrained
    {
        population pop{problem{cec2006{1u}}, 32u, 23u};
        BOOST_CHECK_THROW(nsga3{}.evolve(pop), std::invalid_argument);
    }
    // Population size not a multiple of four, and too small
    {
        population pop{problem{dtlz{1u, 10u, 3u}}, 90u, 23u};
        BOOST_CHECK_THROW((nsga3{1u, 1.0, 30., 0.1, 20., 12u, 0u, true, 32u, false}.evolve(pop)),
                          std::invalid_argument);
    }
    {
        population pop{problem{dtlz{1u, 10u, 3u}}, 4u, 23u};
        BOOST_CHECK_THROW((nsga3{1u, 1.0, 30., 0.1, 20., 1u, 0u, true, 32u, false}.evolve(pop)), std::invalid_argument);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_population_size_rule)
{
    /*  Deb & Jain, Table I: the eight-objective case uses a two layer set of 156
     *  reference directions and a population of exactly 156. Equality between the
     *  two must therefore be accepted, and only a strictly smaller population
     *  rejected.
     */
    const auto directions = detail::generate_reference_directions(8u, 3u, 2u);
    BOOST_REQUIRE_EQUAL(directions.size(), 156u);

    dtlz udp{2u, 10u, 8u};
    nsga3 algo{1u, 1.0, 30., 0.1, 20., 3u, 2u, true, 32u, false};

    population equal_pop{udp, 156u, 23u};
    BOOST_CHECK_NO_THROW(algo.evolve(equal_pop));

    // One reference direction short of the population, and a multiple of four
    population small_pop{udp, 152u, 23u};
    BOOST_CHECK_THROW(algo.evolve(small_pop), std::invalid_argument);

    // Three objectives with twelve divisions: 91 directions, so 92 individuals fit
    BOOST_REQUIRE_EQUAL(detail::generate_reference_directions(3u, 12u, 0u).size(), 91u);
    nsga3 algo3{1u, 1.0, 30., 0.1, 20., 12u, 0u, true, 32u, false};
    population pop92{dtlz{2u, 10u, 3u}, 92u, 23u};
    BOOST_CHECK_NO_THROW(algo3.evolve(pop92));
    population pop88{dtlz{2u, 10u, 3u}, 88u, 23u};
    BOOST_CHECK_THROW(algo3.evolve(pop88), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_reference_point_type)
{
    detail::reference_point rp3(3);
    BOOST_CHECK_EQUAL(rp3.dim(), 3u);
    BOOST_CHECK_EQUAL(rp3[0], 0.0);
    BOOST_CHECK_EQUAL(rp3[1], 0.0);
    BOOST_CHECK_EQUAL(rp3[2], 0.0);

    // reset() drops the generation-specific bookkeeping but keeps the direction
    rp3[1] = 0.5;
    rp3.increment_members();
    rp3.add_candidate(4u, 0.25);
    BOOST_CHECK_EQUAL(rp3.member_count(), 1u);
    BOOST_CHECK_EQUAL(rp3.candidate_count(), 1u);
    rp3.reset();
    BOOST_CHECK_EQUAL(rp3.member_count(), 0u);
    BOOST_CHECK_EQUAL(rp3.candidate_count(), 0u);
    BOOST_CHECK_EQUAL(rp3[1], 0.5);
}

BOOST_AUTO_TEST_CASE(nsga3_n_choose_k)
{
    // Hand values
    BOOST_CHECK_EQUAL(detail::n_choose_k(3u, 2u), 3u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(4u, 2u), 6u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(5u, 3u), 10u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(7u, 5u), 21u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(14u, 12u), 91u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(10u, 3u), 120u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(9u, 2u), 36u);

    // Boundaries
    BOOST_CHECK_EQUAL(detail::n_choose_k(0u, 0u), 1u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(9u, 0u), 1u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(9u, 9u), 1u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(9u, 1u), 9u);
    // k > n is not an error, it is simply zero: the recursive form used to underflow here
    BOOST_CHECK_EQUAL(detail::n_choose_k(5u, 7u), 0u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(0u, 3u), 0u);

    // Large but representable
    BOOST_CHECK_EQUAL(detail::n_choose_k(62u, 31u), 465428353255261088ull);
    // and beyond: reported rather than silently wrapped around
    BOOST_CHECK_THROW(detail::n_choose_k(200u, 100u), std::overflow_error);
    BOOST_CHECK_THROW(detail::n_choose_k(1000u, 500u), std::overflow_error);
}

BOOST_AUTO_TEST_CASE(nsga3_reference_direction_limits)
{
    // The count is the Das and Dennis formula
    BOOST_CHECK_EQUAL(detail::reference_point_count(3u, 12u), 91u);
    BOOST_CHECK_EQUAL(detail::reference_point_count(8u, 3u), 120u);
    BOOST_CHECK_EQUAL(detail::reference_point_count(8u, 2u), 36u);

    // Degenerate arguments are rejected rather than dividing by zero
    BOOST_CHECK_THROW(detail::reference_point_count(3u, 0u), std::invalid_argument);
    BOOST_CHECK_THROW(detail::reference_point_count(0u, 3u), std::invalid_argument);
    BOOST_CHECK_THROW(detail::generate_uniform_reference_points(3u, 0u), std::invalid_argument);

    /*  A layer which would not fit in memory is refused before anything is
     *  allocated. Fifteen objectives with thirty divisions is about 1.1e11
     *  directions.
     */
    BOOST_CHECK(detail::reference_point_count(15u, 30u) > detail::max_reference_directions);
    BOOST_CHECK_THROW(detail::generate_uniform_reference_points(15u, 30u), std::invalid_argument);
    BOOST_CHECK_THROW(detail::generate_reference_directions(15u, 30u, 2u), std::invalid_argument);
    // and one which is merely large is still built
    BOOST_CHECK_NO_THROW(detail::generate_uniform_reference_points(3u, 12u));
}

BOOST_AUTO_TEST_CASE(nsga3_reference_points_exact)
{
    /*  The Das and Dennis enumeration is easy to get subtly wrong, so pin the exact
     *  point sets, in generation order, for the smallest cases.
     */
    auto rp_2_2 = detail::generate_uniform_reference_points(2u, 2u);
    const std::vector<std::vector<double>> expected_2_2{{0.0, 1.0}, {0.5, 0.5}, {1.0, 0.0}};
    BOOST_REQUIRE_EQUAL(rp_2_2.size(), expected_2_2.size());
    for (std::size_t i = 0u; i < expected_2_2.size(); ++i) {
        BOOST_REQUIRE_EQUAL(rp_2_2[i].dim(), expected_2_2[i].size());
        for (std::size_t j = 0u; j < expected_2_2[i].size(); ++j) {
            BOOST_CHECK_SMALL(rp_2_2[i][j] - expected_2_2[i][j], 1e-12);
        }
    }

    // One division per objective yields exactly the canonical axis directions
    auto rp_3_1 = detail::generate_uniform_reference_points(3u, 1u);
    const std::vector<std::vector<double>> expected_3_1{{0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}};
    BOOST_REQUIRE_EQUAL(rp_3_1.size(), expected_3_1.size());
    for (std::size_t i = 0u; i < expected_3_1.size(); ++i) {
        for (std::size_t j = 0u; j < expected_3_1[i].size(); ++j) {
            BOOST_CHECK_SMALL(rp_3_1[i][j] - expected_3_1[i][j], 1e-12);
        }
    }

    auto rp_3_2 = detail::generate_uniform_reference_points(3u, 2u);
    const std::vector<std::vector<double>> expected_3_2{{0.0, 0.0, 1.0}, {0.0, 0.5, 0.5}, {0.0, 1.0, 0.0},
                                                        {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}, {1.0, 0.0, 0.0}};
    BOOST_REQUIRE_EQUAL(rp_3_2.size(), expected_3_2.size());
    for (std::size_t i = 0u; i < expected_3_2.size(); ++i) {
        for (std::size_t j = 0u; j < expected_3_2[i].size(); ++j) {
            BOOST_CHECK_SMALL(rp_3_2[i][j] - expected_3_2[i][j], 1e-12);
        }
    }

    // Every generated point is on the unit simplex: non-negative and summing to one
    for (const auto &set : {rp_2_2, rp_3_1, rp_3_2}) {
        for (const auto &p : set) {
            for (double c : p.get_coeffs()) {
                BOOST_CHECK(std::isfinite(c));
                BOOST_CHECK(c >= 0.0);
            }
            BOOST_CHECK_CLOSE(direction_sum(p), 1.0, 1e-8);
        }
    }

    /*  Table I of Deb & Jain: three objectives with twelve divisions give 91
     *  directions. The count must agree with the Das and Dennis formula in general.
     */
    auto rp_3_12 = detail::generate_uniform_reference_points(3u, 12u);
    BOOST_CHECK_EQUAL(rp_3_12.size(), 91u);
    for (const auto &p : rp_3_12) {
        BOOST_CHECK_CLOSE(direction_sum(p), 1.0, 1e-8);
    }
    BOOST_CHECK_EQUAL(detail::generate_uniform_reference_points(5u, 6u).size(), 210u);

    const std::vector<std::pair<std::size_t, std::size_t>> cases{{2, 1}, {2, 2}, {3, 1}, {3, 2},
                                                                 {3, 3}, {5, 4}, {8, 2}, {10, 3}};
    for (const auto &c : cases) {
        BOOST_CHECK_EQUAL(detail::generate_uniform_reference_points(c.first, c.second).size(),
                          detail::n_choose_k(c.first + c.second - 1u, c.second));
    }

    // Generation is deterministic: the same arguments give the same sequence
    BOOST_CHECK(detail::generate_uniform_reference_points(4u, 3u)[7u].get_coeffs()
                == detail::generate_uniform_reference_points(4u, 3u)[7u].get_coeffs());
}

BOOST_AUTO_TEST_CASE(nsga3_two_layer_reference_directions)
{
    /*  Deb & Jain, Section V: for eight objectives the boundary layer uses p = 3,
     *  giving 120 directions, and the inside layer uses p = 2, giving 36, for a
     *  total of H = 156 as reported in their Table I.
     */
    const std::size_t nobj = 8u;
    auto outer = detail::generate_uniform_reference_points(nobj, 3u);
    auto inner_raw = detail::generate_uniform_reference_points(nobj, 2u);
    BOOST_REQUIRE_EQUAL(outer.size(), 120u);
    BOOST_REQUIRE_EQUAL(inner_raw.size(), 36u);

    auto combined = detail::generate_reference_directions(nobj, 3u, 2u);
    BOOST_REQUIRE_EQUAL(combined.size(), 156u);

    // A zero inner division count leaves the outer layer alone
    auto outer_only = detail::generate_reference_directions(nobj, 3u, 0u);
    BOOST_REQUIRE_EQUAL(outer_only.size(), 120u);
    for (std::size_t i = 0u; i < outer.size(); ++i) {
        BOOST_CHECK(outer_only[i].get_coeffs() == outer[i].get_coeffs());
    }

    // The outer layer is the first 120 directions, in its own generation order
    for (std::size_t i = 0u; i < outer.size(); ++i) {
        BOOST_CHECK(combined[i].get_coeffs() == outer[i].get_coeffs());
    }

    /*  The inner layer follows, each coordinate mapped through (c + 1/M)/2. The
     *  transformation halves the layer about the centroid of the simplex, so the
     *  coordinates still sum to one.
     */
    const double centre = 1.0 / static_cast<double>(nobj);
    for (std::size_t i = 0u; i < inner_raw.size(); ++i) {
        const auto &transformed = combined[outer.size() + i];
        BOOST_REQUIRE_EQUAL(transformed.dim(), nobj);
        for (std::size_t j = 0u; j < nobj; ++j) {
            BOOST_CHECK_SMALL(transformed[j] - (inner_raw[i][j] + centre) / 2.0, 1e-12);
        }
    }

    // Both layers lie on the unit simplex and every coordinate is finite and non-negative
    for (std::size_t i = 0u; i < combined.size(); ++i) {
        for (double c : combined[i].get_coeffs()) {
            BOOST_CHECK(std::isfinite(c));
            BOOST_CHECK(c >= 0.0);
        }
        BOOST_CHECK_CLOSE(direction_sum(combined[i]), 1.0, 1e-8);
    }

    /*  The inner layer is strictly interior: with p = 2 over eight objectives its
     *  coordinates can only be 0.0625, 0.3125 or 0.5625, none of which is on the
     *  boundary of the simplex.
     */
    for (std::size_t i = outer.size(); i < combined.size(); ++i) {
        for (double c : combined[i].get_coeffs()) {
            BOOST_CHECK(c > 0.0);
            const bool expected
                = std::abs(c - 0.0625) < 1e-12 || std::abs(c - 0.3125) < 1e-12 || std::abs(c - 0.5625) < 1e-12;
            BOOST_CHECK(expected);
        }
    }

    // No direction is repeated
    for (std::size_t i = 0u; i < combined.size(); ++i) {
        for (std::size_t j = i + 1u; j < combined.size(); ++j) {
            BOOST_CHECK(combined[i].get_coeffs() != combined[j].get_coeffs());
        }
    }

    // Generation is deterministic
    auto again = detail::generate_reference_directions(nobj, 3u, 2u);
    BOOST_REQUIRE_EQUAL(again.size(), combined.size());
    for (std::size_t i = 0u; i < combined.size(); ++i) {
        BOOST_CHECK(again[i].get_coeffs() == combined[i].get_coeffs());
    }

    // The remaining many-objective rows of Table I
    BOOST_CHECK_EQUAL(detail::generate_reference_directions(10u, 3u, 2u).size(), 275u); // 220 + 55
    BOOST_CHECK_EQUAL(detail::generate_reference_directions(15u, 2u, 1u).size(), 135u); // 120 + 15
    // and the single layer rows
    BOOST_CHECK_EQUAL(detail::generate_reference_directions(3u, 12u, 0u).size(), 91u);
    BOOST_CHECK_EQUAL(detail::generate_reference_directions(5u, 6u, 0u).size(), 210u);
}

BOOST_AUTO_TEST_CASE(nsga3_two_layer_duplicate_elimination)
{
    /*  None of the configurations of Table I produces an inner direction which
     *  coincides with an outer one, but such configurations do exist and must not
     *  yield a repeated direction.
     *
     *  Two objectives with four outer divisions place directions at coordinates
     *  0, 1/4, 1/2, 3/4 and 1. One inner division gives the raw directions (0,1) and
     *  (1,0), which the transformation (c + 1/2)/2 maps onto (1/4, 3/4) and
     *  (3/4, 1/4). Both are already on the outer layer, so both are dropped and only
     *  the five outer directions remain.
     */
    auto outer = detail::generate_uniform_reference_points(2u, 4u);
    BOOST_REQUIRE_EQUAL(outer.size(), 5u);
    auto combined = detail::generate_reference_directions(2u, 4u, 1u);
    BOOST_CHECK_EQUAL(combined.size(), 5u);
    for (std::size_t i = 0u; i < outer.size(); ++i) {
        BOOST_CHECK(combined[i].get_coeffs() == outer[i].get_coeffs());
    }

    /*  Three objectives with two outer divisions and two inner ones: the inner
     *  coordinates are 1/6, 5/12 and 2/3, of which none is on the outer grid of 0,
     *  1/2 and 1, so every inner direction is kept.
     */
    auto kept = detail::generate_reference_directions(3u, 2u, 2u);
    BOOST_CHECK_EQUAL(kept.size(), 12u); // 6 + 6
    for (std::size_t i = 0u; i < kept.size(); ++i) {
        for (std::size_t j = i + 1u; j < kept.size(); ++j) {
            BOOST_CHECK(kept[i].get_coeffs() != kept[j].get_coeffs());
        }
    }

    // Whatever the configuration, no direction is ever repeated
    const std::vector<std::vector<std::size_t>> configs{{2u, 4u, 1u}, {2u, 4u, 2u}, {3u, 4u, 2u},
                                                        {3u, 6u, 3u}, {4u, 3u, 3u}, {5u, 4u, 2u}};
    for (const auto &c : configs) {
        auto set = detail::generate_reference_directions(c[0], c[1], c[2]);
        BOOST_CHECK(!set.empty());
        for (std::size_t i = 0u; i < set.size(); ++i) {
            BOOST_CHECK_CLOSE(direction_sum(set[i]), 1.0, 1e-8);
            for (std::size_t j = i + 1u; j < set.size(); ++j) {
                BOOST_CHECK(set[i].get_coeffs() != set[j].get_coeffs());
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga3_perpendicular_distance)
{
    // A point on an axis is a unit distance from an orthogonal reference direction
    BOOST_CHECK_CLOSE(detail::perpendicular_distance({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}), 1.0, 1e-8);
    // A point lying on the reference ray is at zero distance, wherever it sits along it
    BOOST_CHECK_SMALL(detail::perpendicular_distance({1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}), 1e-12);
    BOOST_CHECK_SMALL(detail::perpendicular_distance({1.0, 1.0, 1.0}, {4.0, 4.0, 4.0}), 1e-12);
    // The projection of (3,4) onto the first axis leaves the second component
    BOOST_CHECK_CLOSE(detail::perpendicular_distance({1.0, 0.0}, {3.0, 4.0}), 4.0, 1e-8);
    BOOST_CHECK_CLOSE(detail::perpendicular_distance({0.5, 0.5}, {1.0, 0.0}), std::sqrt(0.5), 1e-8);

    /*  Only the direction of the reference point matters, not its length: the
     *  reference set is generated on the simplex but is used as a set of rays.
     */
    BOOST_CHECK_CLOSE(detail::perpendicular_distance({2.0, 0.0}, {3.0, 4.0}),
                      detail::perpendicular_distance({1.0, 0.0}, {3.0, 4.0}), 1e-8);
    BOOST_CHECK_CLOSE(detail::perpendicular_distance({0.25, 0.25}, {1.0, 0.0}),
                      detail::perpendicular_distance({0.5, 0.5}, {1.0, 0.0}), 1e-8);

    // The origin projects onto every ray
    BOOST_CHECK_SMALL(detail::perpendicular_distance({0.5, 0.5}, {0.0, 0.0}), 1e-12);
}

BOOST_AUTO_TEST_CASE(nsga3_achievement_scalarization)
{
    // With unit weights the ASF degenerates to the largest component
    BOOST_CHECK_CLOSE(detail::achievement({1.0, 2.0, 3.0}, {1.0, 1.0, 1.0}), 3.0, 1e-8);
    BOOST_CHECK_CLOSE(detail::achievement({5.0, 2.0, 3.0}, {1.0, 1.0, 1.0}), 5.0, 1e-8);

    /*  Weights below the 1e-5 floor are raised to it, so the axis direction used by
     *  find_extreme_points penalises every off-axis component by 1e5.
     */
    BOOST_CHECK_CLOSE(detail::achievement({1.0, 2.0, 3.0}, {1.0, 1e-6, 1e-6}), 3.0e5, 1e-8);
    // A zero weight cannot divide by zero: it is floored just the same
    BOOST_CHECK_CLOSE(detail::achievement({1.0, 2.0, 3.0}, {1.0, 0.0, 0.0}), 3.0e5, 1e-8);
    BOOST_CHECK(std::isfinite(detail::achievement({1.0, 0.0}, {0.0, 0.0})));

    // A point exactly on the axis is not penalised at all
    BOOST_CHECK_CLOSE(detail::achievement({4.0, 0.0, 0.0}, {1.0, 1e-6, 1e-6}), 4.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(nsga3_reference_point_candidates)
{
    detail::reference_point rp(2);
    detail::random_engine_type reng(7u);

    BOOST_CHECK_EQUAL(rp.candidate_count(), 0u);
    BOOST_CHECK_EQUAL(rp.member_count(), 0u);
    // Nothing to select from an empty reference point
    BOOST_CHECK(!rp.select_member(reng).has_value());
    BOOST_CHECK(!rp.nearest_candidate().has_value());

    rp.add_candidate(7u, 0.5);
    rp.add_candidate(3u, 0.1);
    rp.add_candidate(9u, 0.9);
    BOOST_CHECK_EQUAL(rp.candidate_count(), 3u);

    // The nearest candidate is the one with the smallest perpendicular distance
    BOOST_REQUIRE(rp.nearest_candidate().has_value());
    BOOST_CHECK_EQUAL(rp.nearest_candidate().value(), 3u);

    /*  Section IV.E: a reference point with no members yet takes its closest
     *  candidate, which involves no random draw at all.
     */
    auto chosen = rp.select_member(reng);
    BOOST_REQUIRE(chosen.has_value());
    BOOST_CHECK_EQUAL(chosen.value(), 3u);

    // With at least one member the choice is random, but confined to the candidates
    rp.increment_members();
    BOOST_CHECK_EQUAL(rp.member_count(), 1u);
    detail::random_engine_type reng_a(11u), reng_b(11u);
    auto random_a = rp.select_member(reng_a);
    auto random_b = rp.select_member(reng_b);
    BOOST_REQUIRE(random_a.has_value());
    BOOST_REQUIRE(random_b.has_value());
    BOOST_CHECK_EQUAL(random_a.value(), random_b.value()); // same seed, same choice
    BOOST_CHECK(random_a.value() == 3u || random_a.value() == 7u || random_a.value() == 9u);

    // Removing a candidate shifts the nearest one
    rp.remove_candidate(3u);
    BOOST_CHECK_EQUAL(rp.candidate_count(), 2u);
    BOOST_REQUIRE(rp.nearest_candidate().has_value());
    BOOST_CHECK_EQUAL(rp.nearest_candidate().value(), 7u);

    // Removing an index which is not a candidate is a no-op
    rp.remove_candidate(42u);
    BOOST_CHECK_EQUAL(rp.candidate_count(), 2u);

    rp.decrement_members();
    BOOST_CHECK_EQUAL(rp.member_count(), 0u);
}

BOOST_AUTO_TEST_CASE(nsga3_association_golden)
{
    /*  Two objectives and two divisions give the three reference directions
     *  (0,1), (1/2,1/2) and (1,0). Each normalized point below has a hand computed
     *  nearest direction and distance.
     */
    auto rps = detail::generate_uniform_reference_points(2u, 2u);
    BOOST_REQUIRE_EQUAL(rps.size(), 3u);

    const std::vector<std::vector<double>> norm_objs{{0.0, 1.0}, {1.0, 0.0}, {0.5, 0.5}, {0.9, 0.1}};
    const std::vector<std::size_t> expected_nearest{0u, 2u, 1u, 2u};
    const std::vector<double> expected_distance{0.0, 0.0, 0.0, 0.1};

    for (std::size_t i = 0u; i < norm_objs.size(); ++i) {
        std::size_t nearest = 0u;
        double min_dist = std::numeric_limits<double>::max();
        for (std::size_t p = 0u; p < rps.size(); ++p) {
            const double dist = detail::perpendicular_distance(rps[p].get_coeffs(), norm_objs[i]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = p;
            }
        }
        BOOST_CHECK_EQUAL(nearest, expected_nearest[i]);
        BOOST_CHECK_SMALL(min_dist - expected_distance[i], 1e-9);
    }

    /*  Individuals of the earlier fronts become members of their nearest reference
     *  point; only those of the last front become candidates for niching.
     */
    const std::vector<std::vector<pop_size_t>> fronts{{0u, 1u}, {2u, 3u}};
    detail::associate_with_reference_points(rps, norm_objs, fronts);

    BOOST_CHECK_EQUAL(rps[0].member_count(), 1u); // individual 0
    BOOST_CHECK_EQUAL(rps[1].member_count(), 0u);
    BOOST_CHECK_EQUAL(rps[2].member_count(), 1u); // individual 1
    BOOST_CHECK_EQUAL(rps[0].candidate_count(), 0u);
    BOOST_CHECK_EQUAL(rps[1].candidate_count(), 1u); // individual 2
    BOOST_CHECK_EQUAL(rps[2].candidate_count(), 1u); // individual 3
    BOOST_REQUIRE(rps[1].nearest_candidate().has_value());
    BOOST_CHECK_EQUAL(rps[1].nearest_candidate().value(), 2u);
    BOOST_REQUIRE(rps[2].nearest_candidate().has_value());
    BOOST_CHECK_EQUAL(rps[2].nearest_candidate().value(), 3u);

    /*  A single front is the last front, so every individual becomes a candidate
     *  and none becomes a member.
     */
    auto single = detail::generate_uniform_reference_points(2u, 2u);
    const std::vector<std::vector<pop_size_t>> one_front{{0u, 1u, 2u, 3u}};
    detail::associate_with_reference_points(single, norm_objs, one_front);
    std::size_t total_members = 0u, total_candidates = 0u;
    for (const auto &rp : single) {
        total_members += rp.member_count();
        total_candidates += rp.candidate_count();
    }
    BOOST_CHECK_EQUAL(total_members, 0u);
    BOOST_CHECK_EQUAL(total_candidates, norm_objs.size());
}

BOOST_AUTO_TEST_CASE(nsga3_niching_tie_breaking)
{
    detail::random_engine_type reng(1234u);

    // A unique least crowded reference point is returned without consulting the engine
    auto rps = detail::generate_uniform_reference_points(2u, 2u);
    BOOST_REQUIRE_EQUAL(rps.size(), 3u);
    rps[0].increment_members();
    rps[0].increment_members();
    rps[1].increment_members();
    rps[2].increment_members();
    rps[2].increment_members();
    rps[2].increment_members();
    BOOST_CHECK_EQUAL(detail::identify_niche_point(rps, reng), 1u);

    /*  Under a tie the choice is random, so it is only pinned down to the minimal
     *  set. Two identically seeded engines must still agree: this is what makes a
     *  whole run reproducible.
     */
    auto tied = detail::generate_uniform_reference_points(3u, 2u);
    BOOST_REQUIRE_EQUAL(tied.size(), 6u);
    detail::random_engine_type reng_a(99u), reng_b(99u);
    const std::size_t pick_a = detail::identify_niche_point(tied, reng_a);
    const std::size_t pick_b = detail::identify_niche_point(tied, reng_b);
    BOOST_CHECK_EQUAL(pick_a, pick_b);
    BOOST_CHECK(pick_a < tied.size());

    // Only reference points at the minimum member count are eligible
    for (std::size_t i = 0u; i < tied.size(); ++i) {
        if (i != 2u) {
            tied[i].increment_members();
        }
    }
    BOOST_CHECK_EQUAL(detail::identify_niche_point(tied, reng), 2u);
}

BOOST_AUTO_TEST_CASE(nsga3_selection_golden)
{
    /*  Golden environmental selection over fixed objective vectors. The fixtures
     *  are chosen so that the outcome does not depend on how ties are broken:
     *  std::uniform_int_distribution is not specified to map engine output to
     *  values identically across standard library implementations, so an index
     *  drawn from a set of two or more cannot be asserted portably.
     */
    detail::random_engine_type reng(17u);
    const auto directions = detail::generate_reference_directions(2u, 2u, 0u);

    /*  Fixture A: indices 0-3 are the first front, 4 is the whole second front and
     *  5 is dominated by 4. One slot is left after the first front and exactly one
     *  candidate can fill it, so the result is completely determined.
     */
    const std::vector<vector_double> objs_a{{0.0, 1.0}, {0.1, 0.9}, {0.5, 0.5}, {0.45, 0.55}, {0.6, 0.6}, {2.0, 0.6}};
    auto fronts_a = std::get<0>(fast_non_dominated_sorting(objs_a));
    BOOST_REQUIRE_EQUAL(fronts_a.size(), 3u);
    BOOST_REQUIRE_EQUAL(fronts_a[0].size(), 4u);

    auto next_a = detail::nsga3_selection(objs_a, 5u, directions, nullptr, nullptr, reng);
    const std::vector<pop_size_t> expected_a{0u, 1u, 2u, 3u, 4u};
    BOOST_CHECK(next_a == expected_a);

    /*  Fixture B: the second front is absorbed whole, so every individual survives
     *  whatever order the niching visits the reference points in.
     */
    const std::vector<vector_double> objs_b{{0.0, 1.0}, {0.1, 0.9}, {0.5, 0.5}, {1.0, 0.0}, {0.6, 0.6}, {1.1, 0.1}};
    auto next_b = detail::nsga3_selection(objs_b, 6u, directions, nullptr, nullptr, reng);
    std::vector<pop_size_t> sorted_b{next_b};
    std::sort(sorted_b.begin(), sorted_b.end());
    const std::vector<pop_size_t> expected_b{0u, 1u, 2u, 3u, 4u, 5u};
    BOOST_CHECK(sorted_b == expected_b);

    /*  Fixture C: two candidates compete for one slot, so which one survives is a
     *  random tie-break. The structural guarantees still hold exactly, and the
     *  choice must be reproducible for a given engine state.
     */
    detail::random_engine_type reng_c1(5u), reng_c2(5u);
    auto next_c1 = detail::nsga3_selection(objs_b, 5u, directions, nullptr, nullptr, reng_c1);
    auto next_c2 = detail::nsga3_selection(objs_b, 5u, directions, nullptr, nullptr, reng_c2);
    BOOST_CHECK(next_c1 == next_c2); // same seed, same survivors

    BOOST_CHECK_EQUAL(next_c1.size(), 5u);
    std::vector<pop_size_t> sorted_c{next_c1};
    std::sort(sorted_c.begin(), sorted_c.end());
    BOOST_CHECK(std::unique(sorted_c.begin(), sorted_c.end()) == sorted_c.end()); // no duplicates
    for (auto idx : next_c1) {
        BOOST_CHECK(idx < objs_b.size());
    }
    // The whole first front survives, and the last slot comes from the second front
    auto fronts_b = std::get<0>(fast_non_dominated_sorting(objs_b));
    BOOST_REQUIRE_EQUAL(fronts_b[0].size(), 4u);
    for (auto idx : fronts_b[0]) {
        BOOST_CHECK(std::find(next_c1.begin(), next_c1.end(), idx) != next_c1.end());
    }
    std::size_t from_last_front = 0u;
    for (auto idx : fronts_b[1]) {
        from_last_front += (std::find(next_c1.begin(), next_c1.end(), idx) != next_c1.end()) ? 1u : 0u;
    }
    BOOST_CHECK_EQUAL(from_last_front, 1u);

    // Degenerate arguments are rejected
    BOOST_CHECK_THROW(detail::nsga3_selection(objs_b, 0u, directions, nullptr, nullptr, reng), std::invalid_argument);
    BOOST_CHECK_THROW(detail::nsga3_selection(objs_b, 7u, directions, nullptr, nullptr, reng), std::invalid_argument);
    BOOST_CHECK_THROW(detail::nsga3_selection(objs_b, 5u, {}, nullptr, nullptr, reng), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_selection_preserves_size)
{
    /*  Environmental selection over a larger, structured set: whatever the fronts
     *  look like, exactly N_pop distinct in-range indices come back.
     */
    dtlz udp{2u, 10u, 3u};
    population pop{udp, 64u, 23u};
    detail::random_engine_type reng(3u);
    const auto directions = detail::generate_reference_directions(3u, 4u, 0u);

    const auto objs = pop.get_f();
    for (pop_size_t n_pop : {8u, 16u, 32u, 63u}) {
        auto next = detail::nsga3_selection(objs, n_pop, directions, nullptr, nullptr, reng);
        BOOST_CHECK_EQUAL(next.size(), n_pop);
        std::vector<pop_size_t> sorted_next{next};
        std::sort(sorted_next.begin(), sorted_next.end());
        BOOST_CHECK(std::unique(sorted_next.begin(), sorted_next.end()) == sorted_next.end());
        for (auto idx : next) {
            BOOST_CHECK(idx < objs.size());
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_translate_objectives)
{
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 0u, true, 32u, false};

    pop = nsga3_alg.evolve(pop);
    auto p0_obj = pop.get_f();
    std::vector<pop_size_t> all(p0_obj.size());
    std::iota(all.begin(), all.end(), pop_size_t(0));
    auto ideal_point = detail::nsga3_compute_ideal(p0_obj, all, nullptr);
    auto translated_objectives = detail::nsga3_translate_objectives(p0_obj, ideal_point);

    BOOST_REQUIRE_EQUAL(translated_objectives.size(), p0_obj.size());
    BOOST_REQUIRE_EQUAL(ideal_point.size(), udp.get_nobj());

    // The ideal point is the componentwise minimum of the objectives
    for (std::size_t obj = 0u; obj < ideal_point.size(); ++obj) {
        double column_min = std::numeric_limits<double>::max();
        for (const auto &f : p0_obj) {
            column_min = std::min(column_min, f[obj]);
        }
        BOOST_CHECK_EQUAL(ideal_point[obj], column_min);
    }

    // Translation is an exact componentwise shift by that point
    for (std::size_t i = 0u; i < p0_obj.size(); ++i) {
        BOOST_REQUIRE_EQUAL(translated_objectives[i].size(), ideal_point.size());
        for (std::size_t obj = 0u; obj < ideal_point.size(); ++obj) {
            BOOST_CHECK_EQUAL(translated_objectives[i][obj], p0_obj[i][obj] - ideal_point[obj]);
            // Which leaves the whole population in the non-negative orthant
            BOOST_CHECK(translated_objectives[i][obj] >= 0.0);
        }
    }

    // and puts the origin at the ideal point: every objective attains zero
    for (std::size_t obj = 0u; obj < ideal_point.size(); ++obj) {
        double column_min = std::numeric_limits<double>::max();
        for (const auto &row : translated_objectives) {
            column_min = std::min(column_min, row[obj]);
        }
        BOOST_CHECK_SMALL(column_min, 1e-12);
    }

    // An empty selected set is rejected rather than silently producing an empty ideal point
    BOOST_CHECK_THROW(detail::nsga3_compute_ideal(p0_obj, {}, nullptr), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_test_gaussian_elimination)
{
    // Verify correctness of simple system
    std::vector<std::vector<double>> A(3);
    std::vector<double> b = {1.0, 1.0, 1.0};

    A[0] = {-1, 1, 2};
    A[1] = {2, 0, -3};
    A[2] = {5, 1, -2};

    auto x = detail::gaussian_elimination(A, b);
    BOOST_REQUIRE(x.has_value());
    BOOST_CHECK_CLOSE((*x)[0], -0.4, 1e-8);
    BOOST_CHECK_CLOSE((*x)[1], 1.8, 1e-8);
    BOOST_CHECK_CLOSE((*x)[2], -0.6, 1e-8);

    /*  A zero leading pivot is not an error: partial pivoting selects the largest
     *  available pivot in each column, so this non-singular system is solvable.
     */
    std::vector<std::vector<double>> pivoted{{0.0, 2.0, 1.0}, {1.0, 0.0, 3.0}, {2.0, 1.0, 0.0}};
    auto xp = detail::gaussian_elimination(pivoted, b);
    BOOST_REQUIRE(xp.has_value());
    for (std::size_t i = 0u; i < pivoted.size(); ++i) {
        double residual = 0.0;
        for (std::size_t j = 0u; j < pivoted[i].size(); ++j) {
            residual += pivoted[i][j] * (*xp)[j];
        }
        BOOST_CHECK_CLOSE(residual, b[i], 1e-8);
    }

    // An exactly singular system is reported, not thrown: the third row is row0 + row1
    std::vector<std::vector<double>> singular{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {5.0, 7.0, 9.0}};
    BOOST_CHECK_NO_THROW((detail::gaussian_elimination(singular, b)));
    BOOST_CHECK(!detail::gaussian_elimination(singular, b).has_value());

    /*  A nearly singular system is caught by the scale-aware tolerance: the first
     *  two rows differ by an amount well below eps*N*max|A_ij|.
     */
    std::vector<std::vector<double>> near_singular{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0 + 1e-15}, {1.0, 2.0, 3.0}};
    BOOST_CHECK(!detail::gaussian_elimination(near_singular, b).has_value());

    // The same system, perturbed well above the tolerance, remains solvable
    std::vector<std::vector<double>> conditioned{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0 + 1e-6}, {1.0, 2.0, 3.0}};
    BOOST_CHECK(detail::gaussian_elimination(conditioned, b).has_value());

    // A non-finite entry is a failure, not an exception
    std::vector<std::vector<double>> non_finite{
        {std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0}, {1.0, 1.0, 2.0}, {1.0, 2.0, 3.0}};
    BOOST_CHECK(!detail::gaussian_elimination(non_finite, b).has_value());

    // Dimensions are validated
    std::vector<std::vector<double>> empty_matrix;
    BOOST_CHECK_THROW((detail::gaussian_elimination(empty_matrix, b)), std::invalid_argument);
    std::vector<std::vector<double>> non_square{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    BOOST_CHECK_THROW((detail::gaussian_elimination(non_square, b)), std::invalid_argument);
    std::vector<double> short_b{1.0, 1.0};
    BOOST_CHECK_THROW((detail::gaussian_elimination(A, short_b)), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_normalization_over_selected_set)
{
    /*  Deb & Jain Algorithm 2 operates on S_t, the accepted fronts together with the
     *  splitting front, and not on the first non-dominated front alone. The fixture
     *  below separates the two: front 0 is {(1,4), (4,1)}, front 1 is the single
     *  dominated point {(5,5)} and front 2 is {(9,9)}.
     */
    const std::vector<vector_double> objs{{1.0, 4.0}, {4.0, 1.0}, {5.0, 5.0}, {9.0, 9.0}};
    auto fronts = std::get<0>(fast_non_dominated_sorting(objs));
    BOOST_REQUIRE_EQUAL(fronts.size(), 3u);
    BOOST_REQUIRE_EQUAL(fronts[0].size(), 2u);
    BOOST_REQUIRE_EQUAL(fronts[1].size(), 1u);

    const std::vector<pop_size_t> selected{0u, 1u, 2u}; // S_t = F_0 U F_1
    const std::vector<pop_size_t> first_front{0u, 1u};

    // The ideal point is the componentwise minimum over S_t
    auto ideal_point = detail::nsga3_compute_ideal(objs, selected, nullptr);
    BOOST_CHECK_CLOSE(ideal_point[0], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal_point[1], 1.0, 1e-8);

    // Translated objectives are (0,3), (3,0), (4,4) and (8,8)
    auto translated = detail::nsga3_translate_objectives(objs, ideal_point);

    /*  The componentwise maximum over S_t is (4,4) and is attained by the dominated
     *  member of the splitting front. The nadir point of the first front is (3,3),
     *  a materially different answer, and the individual outside S_t must not
     *  contribute at all even though it is the largest of the four.
     */
    auto maxima = detail::nsga3_translated_maxima(translated, selected);
    BOOST_CHECK_CLOSE(maxima[0], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(maxima[1], 4.0, 1e-8);

    auto first_front_maxima = detail::nsga3_translated_maxima(translated, first_front);
    BOOST_CHECK_CLOSE(first_front_maxima[0], 3.0, 1e-8);
    BOOST_CHECK_CLOSE(first_front_maxima[1], 3.0, 1e-8);
    BOOST_CHECK(maxima != first_front_maxima);

    // Duplicate extreme points force the fallback, which must be the maxima over S_t
    const std::vector<std::vector<double>> duplicated{{2.0, 2.0}, {2.0, 2.0}};
    auto intercepts = detail::nsga3_find_intercepts(duplicated, translated, selected);
    BOOST_CHECK_CLOSE(intercepts[0], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 4.0, 1e-8);

    // A singular system takes the same path
    const std::vector<std::vector<double>> singular{{0.0, 0.0}, {0.0, 2.0}};
    BOOST_CHECK_NO_THROW((intercepts = detail::nsga3_find_intercepts(singular, translated, selected)));
    BOOST_CHECK_CLOSE(intercepts[0], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 4.0, 1e-8);

    // As does a solution with a negative component, which has no usable reciprocal
    const std::vector<std::vector<double>> negative{{-1.0, 0.0}, {0.0, 2.0}};
    intercepts = detail::nsga3_find_intercepts(negative, translated, selected);
    BOOST_CHECK_CLOSE(intercepts[0], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 4.0, 1e-8);

    // and a zero solution component, and a non-finite one
    const std::vector<std::vector<double>> zero_solution{{0.0, 1.0}, {0.0, 2.0}};
    intercepts = detail::nsga3_find_intercepts(zero_solution, translated, selected);
    BOOST_CHECK_CLOSE(intercepts[0], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 4.0, 1e-8);

    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<std::vector<double>> non_finite{{inf, 0.0}, {0.0, 2.0}};
    BOOST_CHECK_NO_THROW((intercepts = detail::nsga3_find_intercepts(non_finite, translated, selected)));
    BOOST_CHECK_CLOSE(intercepts[0], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 4.0, 1e-8);

    /*  Two extreme points which differ by more than the duplicate tolerance but
     *  which still leave the system ill conditioned: the solved hyperplane is
     *  degenerate and unusable, so the fallback is taken rather than an intercept
     *  of astronomical magnitude being returned.
     */
    const std::vector<std::vector<double>> ill_conditioned{{1.0, 1.0}, {1.0, 1.0 + 1e-11}};
    intercepts = detail::nsga3_find_intercepts(ill_conditioned, translated, selected);
    BOOST_CHECK_CLOSE(intercepts[0], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 4.0, 1e-8);

    // In every case the normalized values are finite
    for (const auto &ext : {duplicated, singular, negative, zero_solution, non_finite, ill_conditioned}) {
        const auto its = detail::nsga3_find_intercepts(ext, translated, selected);
        const auto norm = detail::nsga3_normalize_objectives(translated, its);
        for (auto idx : selected) {
            for (double value : norm[idx]) {
                BOOST_CHECK(std::isfinite(value));
            }
        }
    }

    // A well conditioned system is solved rather than falling back
    const std::vector<std::vector<double>> spread{{2.0, 0.0}, {0.0, 8.0}};
    intercepts = detail::nsga3_find_intercepts(spread, translated, selected);
    BOOST_CHECK_CLOSE(intercepts[0], 2.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 8.0, 1e-8);

    // An empty selected set is rejected
    BOOST_CHECK_THROW(detail::nsga3_translated_maxima(translated, {}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_test_extreme_point_duplicates)
{
    /*  Translated objectives whose maximum over S_t, (2, 2, 2), differs from the
     *  intercepts the solver produces.
     */
    const std::vector<std::vector<double>> translated{{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}};
    const std::vector<pop_size_t> selected{0u, 1u};

    /*  These extreme points are pairwise distinct, but every pair shares at least
     *  one coordinate. Comparing coordinates individually misclassifies them as
     *  duplicates and skips the solver; comparing complete vectors does not.
     */
    std::vector<std::vector<double>> distinct{{2.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 8.0}};
    auto intercepts = detail::nsga3_find_intercepts(distinct, translated, selected);
    BOOST_CHECK_CLOSE(intercepts[0], 2.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[2], 8.0, 1e-8);

    // Two identical extreme points do fall back to the maxima over S_t
    std::vector<std::vector<double>> duplicated{{2.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 0.0, 8.0}};
    intercepts = detail::nsga3_find_intercepts(duplicated, translated, selected);
    BOOST_CHECK_CLOSE(intercepts[0], 2.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 2.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[2], 2.0, 1e-8);

    // As do points which differ only within the numerical tolerance
    std::vector<std::vector<double>> near_duplicated{{2.0, 0.0, 0.0}, {2.0 + 1e-15, 1e-16, -1e-16}, {0.0, 0.0, 8.0}};
    intercepts = detail::nsga3_find_intercepts(near_duplicated, translated, selected);
    BOOST_CHECK_CLOSE(intercepts[0], 2.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 2.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[2], 2.0, 1e-8);

    // An extreme point of the wrong dimension also falls back rather than indexing out of range
    std::vector<std::vector<double>> ragged{{2.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 8.0}};
    BOOST_CHECK_NO_THROW((intercepts = detail::nsga3_find_intercepts(ragged, translated, selected)));
    BOOST_CHECK_CLOSE(intercepts[0], 2.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(nsga3_test_degenerate_objective)
{
    /*  A degenerate objective, identical across the population, has zero extent.
     *  The intercept is sanitised to 1.0 so that normalization leaves the
     *  coordinate at zero instead of producing an infinity or a NaN.
     */
    const std::vector<std::vector<double>> degenerate{{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}};
    const std::vector<pop_size_t> selected{0u, 1u, 2u};
    std::vector<std::vector<double>> degenerate_ext{{0.0, 1.0}, {0.0, 1.0}};
    auto degenerate_intercepts = detail::nsga3_find_intercepts(degenerate_ext, degenerate, selected);
    BOOST_CHECK_CLOSE(degenerate_intercepts[0], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(degenerate_intercepts[1], 3.0, 1e-8);
    auto norm_objs = detail::nsga3_normalize_objectives(degenerate, degenerate_intercepts);
    for (const auto &row : norm_objs) {
        BOOST_CHECK_EQUAL(row[0], 0.0);
        for (double value : row) {
            BOOST_CHECK(std::isfinite(value));
        }
    }

    // A whole population collapsed to a single point stays finite too
    const std::vector<std::vector<double>> collapsed{{0.0, 0.0}, {0.0, 0.0}};
    const std::vector<pop_size_t> both{0u, 1u};
    auto collapsed_intercepts = detail::nsga3_find_intercepts({{0.0, 0.0}, {0.0, 0.0}}, collapsed, both);
    for (double intercept : collapsed_intercepts) {
        BOOST_CHECK(std::isfinite(intercept));
        BOOST_CHECK(intercept > 0.0);
    }
    for (const auto &row : detail::nsga3_normalize_objectives(collapsed, collapsed_intercepts)) {
        for (double value : row) {
            BOOST_CHECK(std::isfinite(value));
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_normalize_nonzero_ideal)
{
    /*  Ideal point (2, 5, -1), well away from the origin. The first three points
     *  are mutually non-dominated and the fourth is dominated by all of them.
     */
    const std::vector<vector_double> objs{{2.0, 9.0, 3.0}, {6.0, 5.0, 3.0}, {6.0, 9.0, -1.0}, {6.0, 9.0, 3.0}};
    const std::vector<pop_size_t> selected{0u, 1u, 2u, 3u};

    auto ideal_point = detail::nsga3_compute_ideal(objs, selected, nullptr);
    BOOST_CHECK_CLOSE(ideal_point[0], 2.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal_point[1], 5.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal_point[2], -1.0, 1e-8);

    auto translated = detail::nsga3_translate_objectives(objs, ideal_point);
    for (std::size_t obj = 0u; obj < ideal_point.size(); ++obj) {
        double col_min = std::numeric_limits<double>::max();
        for (const auto &row : translated) {
            col_min = std::min(col_min, row[obj]);
        }
        BOOST_CHECK_SMALL(col_min, 1e-12);
    }

    auto ext_points = detail::nsga3_find_extreme_points(selected, translated, ideal_point, nullptr);
    auto intercepts = detail::nsga3_find_intercepts(ext_points, translated, selected);

    /*  The extreme points coincide here, so the fallback is taken. The intercepts
     *  must be the componentwise maximum of the *translated* objectives over S_t,
     *  (4, 4, 4), and not the maximum of the original objectives, (6, 9, 3).
     */
    BOOST_CHECK_CLOSE(intercepts[0], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[2], 4.0, 1e-8);

    // The individual sitting on the intercept vector normalizes to (1, 1, 1)
    auto norm_objs = detail::nsga3_normalize_objectives(translated, intercepts);
    BOOST_CHECK_CLOSE(norm_objs[3][0], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(norm_objs[3][1], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(norm_objs[3][2], 1.0, 1e-8);

    /*  The same ideal point with well separated extreme points, so that the
     *  solver path is exercised: translated extremes (9,0,0), (0,5,0), (0,0,1)
     *  give x = (1/9, 1/5, 1) and therefore intercepts (9, 5, 1).
     */
    const std::vector<vector_double> spread_objs{{11.0, 5.0, -1.0}, {2.0, 10.0, -1.0}, {2.0, 5.0, 0.0}};
    const std::vector<pop_size_t> spread_selected{0u, 1u, 2u};
    auto spread_ideal = detail::nsga3_compute_ideal(spread_objs, spread_selected, nullptr);
    BOOST_CHECK_CLOSE(spread_ideal[0], 2.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_ideal[1], 5.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_ideal[2], -1.0, 1e-8);
    auto spread_translated = detail::nsga3_translate_objectives(spread_objs, spread_ideal);
    auto spread_ext = detail::nsga3_find_extreme_points(spread_selected, spread_translated, spread_ideal, nullptr);
    auto spread_intercepts = detail::nsga3_find_intercepts(spread_ext, spread_translated, spread_selected);
    BOOST_CHECK_CLOSE(spread_intercepts[0], 9.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_intercepts[1], 5.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_intercepts[2], 1.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(nsga3_test_memory_ideal_shift)
{
    const std::vector<pop_size_t> selected{0u, 1u, 2u};

    // Generation 1: the ideal point is (10, 10, 10)
    const std::vector<vector_double> objs1{{19.0, 10.0, 10.0}, {10.0, 15.0, 10.0}, {10.0, 10.0, 11.0}};
    // Generation 2: the ideal point improves to (8, 9, 10)
    const std::vector<vector_double> objs2{{8.0, 30.0, 30.0}, {30.0, 9.0, 30.0}, {30.0, 30.0, 10.0}};

    /*  With memory enabled the running ideal point and the retained extreme
     *  points persist across generations; nsga3 owns these two buffers.
     */
    std::vector<double> running_ideal;
    std::vector<std::vector<double>> retained_extremes;

    auto ideal1 = detail::nsga3_compute_ideal(objs1, selected, &running_ideal);
    BOOST_CHECK_CLOSE(ideal1[0], 10.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal1[1], 10.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal1[2], 10.0, 1e-8);
    auto translated1 = detail::nsga3_translate_objectives(objs1, ideal1);
    auto ext1 = detail::nsga3_find_extreme_points(selected, translated1, ideal1, &retained_extremes);
    // Translated extremes of generation 1: (9,0,0), (0,5,0), (0,0,1)
    BOOST_CHECK_CLOSE(ext1[0][0], 9.0, 1e-8);
    BOOST_CHECK_CLOSE(ext1[1][1], 5.0, 1e-8);
    BOOST_CHECK_CLOSE(ext1[2][2], 1.0, 1e-8);

    // The running ideal point is the elementwise minimum over both generations
    auto ideal2 = detail::nsga3_compute_ideal(objs2, selected, &running_ideal);
    BOOST_CHECK_CLOSE(ideal2[0], 8.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal2[1], 9.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal2[2], 10.0, 1e-8);

    auto translated2 = detail::nsga3_translate_objectives(objs2, ideal2);
    auto ext2 = detail::nsga3_find_extreme_points(selected, translated2, ideal2, &retained_extremes);

    /*  Generation 2 has no candidate better than the retained extreme points, so
     *  those are returned, re-expressed in the *current* translated coordinates:
     *  (19,10,10) - (8,9,10) = (11,1,0), and so on. Had the extreme points been
     *  retained in the translated coordinates of generation 1 they would still
     *  read (9,0,0), (0,5,0), (0,0,1) here.
     */
    const std::vector<std::vector<double>> expected_ext2{{11.0, 1.0, 0.0}, {2.0, 6.0, 0.0}, {2.0, 1.0, 1.0}};
    for (std::size_t i = 0u; i < expected_ext2.size(); ++i) {
        for (std::size_t j = 0u; j < expected_ext2[i].size(); ++j) {
            BOOST_CHECK_SMALL(ext2[i][j] - expected_ext2[i][j], 1e-9);
        }
    }

    // A worse generation does not degrade the retained ideal point
    const std::vector<vector_double> objs3{{20.0, 20.0, 20.0}, {21.0, 21.0, 21.0}, {22.0, 22.0, 22.0}};
    auto ideal3 = detail::nsga3_compute_ideal(objs3, selected, &running_ideal);
    BOOST_CHECK_CLOSE(ideal3[0], 8.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal3[1], 9.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal3[2], 10.0, 1e-8);

    /*  Without memory the same generation-2 input depends only on the current
     *  objectives: the extreme point for the first objective is (22, 0, 20).
     */
    auto plain_ideal = detail::nsga3_compute_ideal(objs2, selected, nullptr);
    auto plain_translated = detail::nsga3_translate_objectives(objs2, plain_ideal);
    auto plain_ext = detail::nsga3_find_extreme_points(selected, plain_translated, plain_ideal, nullptr);
    BOOST_CHECK_CLOSE(plain_ext[0][0], 22.0, 1e-8);
    BOOST_CHECK_SMALL(plain_ext[0][1], 1e-12);
    BOOST_CHECK_CLOSE(plain_ext[0][2], 20.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(nsga3_test_find_extreme_points)
{
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 0u, true, 32u, false};

    pop = nsga3_alg.evolve(pop);
    auto objs = pop.get_f();
    const std::size_t nobj = udp.get_nobj();

    /*  The extreme points are sought over the whole of S_t. Here S_t is taken to be
     *  the first two fronts, so that the search set is strictly larger than the
     *  first front alone.
     */
    auto fronts = std::get<0>(fast_non_dominated_sorting(objs));
    BOOST_REQUIRE(fronts.size() >= 1u);
    std::vector<pop_size_t> selected;
    for (std::size_t f = 0u; f < std::min<std::size_t>(2u, fronts.size()); ++f) {
        selected.insert(selected.end(), fronts[f].begin(), fronts[f].end());
    }

    auto ideal_point = detail::nsga3_compute_ideal(objs, selected, nullptr);
    auto translated_objectives = detail::nsga3_translate_objectives(objs, ideal_point);
    auto ext_points = detail::nsga3_find_extreme_points(selected, translated_objectives, ideal_point, nullptr);

    BOOST_REQUIRE_EQUAL(ext_points.size(), nobj);

    for (std::size_t axis = 0u; axis < nobj; ++axis) {
        BOOST_REQUIRE_EQUAL(ext_points[axis].size(), nobj);

        /*  Recompute the achievement scalarization independently: the returned
         *  extreme point must be the member of S_t which minimises it for this
         *  axis, which is the whole contract of the function.
         */
        std::vector<double> weights(nobj, 1e-6);
        weights[axis] = 1.0;
        double best_asf = std::numeric_limits<double>::max();
        for (auto idx : selected) {
            best_asf = std::min(best_asf, detail::achievement(translated_objectives[idx], weights));
        }
        BOOST_CHECK_CLOSE(detail::achievement(ext_points[axis], weights), best_asf, 1e-8);

        // and it must be one of those individuals, not a synthesised point
        bool found = false;
        for (auto idx : selected) {
            if (translated_objectives[idx] == ext_points[axis]) {
                found = true;
                break;
            }
        }
        BOOST_CHECK(found);

        // Extremes come from the translated population, so they are non-negative
        for (double value : ext_points[axis]) {
            BOOST_CHECK(std::isfinite(value));
            BOOST_CHECK(value >= 0.0);
        }
    }

    // An empty selected set is rejected
    BOOST_CHECK_THROW(detail::nsga3_find_extreme_points({}, translated_objectives, ideal_point, nullptr),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_test_find_intercepts)
{
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 0u, true, 32u, false};

    pop = nsga3_alg.evolve(pop);
    auto objs = pop.get_f();
    std::vector<pop_size_t> selected(objs.size());
    std::iota(selected.begin(), selected.end(), pop_size_t(0));

    auto ideal_point = detail::nsga3_compute_ideal(objs, selected, nullptr);
    auto translated_objectives = detail::nsga3_translate_objectives(objs, ideal_point);
    auto ext_points = detail::nsga3_find_extreme_points(selected, translated_objectives, ideal_point, nullptr);

    auto intercepts = detail::nsga3_find_intercepts(ext_points, translated_objectives, selected);
    BOOST_REQUIRE_EQUAL(intercepts.size(), udp.get_nobj());
    // Intercepts are always usable divisors
    for (double intercept : intercepts) {
        BOOST_CHECK(std::isfinite(intercept));
        BOOST_CHECK(intercept > 0.0);
    }
    /*  Each extreme point, expressed in units of the intercepts, stays finite and
     *  positive. The stronger identity below cannot be asserted here because the
     *  fallback legitimately breaks it whenever the extreme points coincide.
     */
    for (const auto &ext : ext_points) {
        double plane = 0.0;
        for (std::size_t j = 0u; j < intercepts.size(); ++j) {
            plane += ext[j] / intercepts[j];
        }
        BOOST_CHECK(std::isfinite(plane));
        BOOST_CHECK(plane > 0.0);
    }

    /*  With well separated extreme points the solver path is taken, and the
     *  intercepts are by construction the axis crossings of the hyperplane through
     *  them: every extreme point then satisfies sum_j ext[j]/intercept[j] == 1.
     */
    const std::vector<std::vector<double>> spread_translated{
        {9.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 1.0}, {3.0, 2.0, 0.5}};
    const std::vector<std::vector<double>> spread_ext{{9.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 1.0}};
    const std::vector<pop_size_t> spread_selected{0u, 1u, 2u, 3u};
    auto spread_intercepts = detail::nsga3_find_intercepts(spread_ext, spread_translated, spread_selected);
    BOOST_CHECK_CLOSE(spread_intercepts[0], 9.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_intercepts[1], 5.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_intercepts[2], 1.0, 1e-8);
    for (const auto &ext : spread_ext) {
        double plane = 0.0;
        for (std::size_t j = 0u; j < spread_intercepts.size(); ++j) {
            plane += ext[j] / spread_intercepts[j];
        }
        BOOST_CHECK_CLOSE(plane, 1.0, 1e-8);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_normalize_objectives)
{
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 0u, true, 32u, false};

    pop = nsga3_alg.evolve(pop);
    auto objs = pop.get_f();
    std::vector<pop_size_t> selected(objs.size());
    std::iota(selected.begin(), selected.end(), pop_size_t(0));

    auto ideal_point = detail::nsga3_compute_ideal(objs, selected, nullptr);
    auto translated_objectives = detail::nsga3_translate_objectives(objs, ideal_point);
    auto ext_points = detail::nsga3_find_extreme_points(selected, translated_objectives, ideal_point, nullptr);
    auto intercepts = detail::nsga3_find_intercepts(ext_points, translated_objectives, selected);
    auto norm_objs = detail::nsga3_normalize_objectives(translated_objectives, intercepts);

    BOOST_REQUIRE_EQUAL(norm_objs.size(), translated_objectives.size());

    // Normalization divides each objective by its own intercept, and nothing else
    for (std::size_t i = 0u; i < translated_objectives.size(); ++i) {
        BOOST_REQUIRE_EQUAL(norm_objs[i].size(), intercepts.size());
        for (std::size_t j = 0u; j < intercepts.size(); ++j) {
            BOOST_CHECK_EQUAL(norm_objs[i][j], translated_objectives[i][j] / intercepts[j]);
            BOOST_CHECK(std::isfinite(norm_objs[i][j]));
            // The translated objectives are non-negative and the intercepts positive
            BOOST_CHECK(norm_objs[i][j] >= 0.0);
        }
    }

    // An individual sitting on the intercept vector lands on the unit hyperplane
    auto on_plane = detail::nsga3_normalize_objectives({intercepts}, intercepts);
    BOOST_REQUIRE_EQUAL(on_plane.size(), 1u);
    for (double value : on_plane[0]) {
        BOOST_CHECK_CLOSE(value, 1.0, 1e-8);
    }

    // An empty input is not an error
    BOOST_CHECK(detail::nsga3_normalize_objectives({}, intercepts).empty());
}

BOOST_AUTO_TEST_CASE(nsga3_reproducibility_same_seed)
{
    dtlz udp{1u, 10u, 3u};

    for (bool random_mating : {true, false}) {
        population pop_a{udp, 52u, 23u};
        population pop_b{udp, 52u, 23u};
        nsga3 alg_a{5u, 1.00, 30., 0.10, 20., 5u, 0u, random_mating, 42u, false};
        nsga3 alg_b{5u, 1.00, 30., 0.10, 20., 5u, 0u, random_mating, 42u, false};
        alg_a.set_verbosity(1u);
        alg_b.set_verbosity(1u);

        pop_a = alg_a.evolve(pop_a);
        pop_b = alg_b.evolve(pop_b);

        BOOST_CHECK(pop_a.get_x() == pop_b.get_x());
        BOOST_CHECK(pop_a.get_f() == pop_b.get_f());
        BOOST_CHECK(alg_a.get_log() == alg_b.get_log());
    }
}

BOOST_AUTO_TEST_CASE(nsga3_mating_modes)
{
    /*  Both mating schemes are deterministic under a fixed seed, and the tournament
     *  really is a different mechanism from the random pairing of Section IV-F: the
     *  two must not silently coincide.
     */
    dtlz udp{2u, 10u, 3u};

    population pop_random{udp, 52u, 23u};
    nsga3 alg_random{5u, 1.00, 30., 0.10, 20., 5u, 0u, true, 42u, false};
    pop_random = alg_random.evolve(pop_random);

    population pop_tournament{udp, 52u, 23u};
    nsga3 alg_tournament{5u, 1.00, 30., 0.10, 20., 5u, 0u, false, 42u, false};
    pop_tournament = alg_tournament.evolve(pop_tournament);

    BOOST_CHECK(pop_random.get_f() != pop_tournament.get_f());

    // Each mode is reproducible on its own
    population pop_tournament_again{udp, 52u, 23u};
    nsga3 alg_tournament_again{5u, 1.00, 30., 0.10, 20., 5u, 0u, false, 42u, false};
    pop_tournament_again = alg_tournament_again.evolve(pop_tournament_again);
    BOOST_CHECK(pop_tournament.get_x() == pop_tournament_again.get_x());
    BOOST_CHECK(pop_tournament.get_f() == pop_tournament_again.get_f());

    // Neither mode changes how many offspring are produced and evaluated
    BOOST_CHECK_EQUAL(pop_random.get_problem().get_fevals(), 52u + 5u * 52u);
    BOOST_CHECK_EQUAL(pop_tournament.get_problem().get_fevals(), 52u + 5u * 52u);
}

BOOST_AUTO_TEST_CASE(nsga3_bfe_matches_scalar)
{
    /*  The offspring population is generated in full before any of it is evaluated,
     *  so a batch evaluator which reproduces the values of the problem must lead to
     *  exactly the same evolution as the scalar path.
     */
    dtlz udp{2u, 10u, 3u};

    for (bool random_mating : {true, false}) {
        population pop_scalar{udp, 52u, 23u};
        const auto fevals0 = pop_scalar.get_problem().get_fevals();
        nsga3 alg_scalar{5u, 1.00, 30., 0.10, 20., 5u, 0u, random_mating, 42u, false};
        pop_scalar = alg_scalar.evolve(pop_scalar);
        const auto scalar_fevals = pop_scalar.get_problem().get_fevals() - fevals0;
        BOOST_CHECK_EQUAL(scalar_fevals, 5u * 52u);

        population pop_batch{udp, 52u, 23u};
        const auto batch_fevals0 = pop_batch.get_problem().get_fevals();
        nsga3 alg_batch{5u, 1.00, 30., 0.10, 20., 5u, 0u, random_mating, 42u, false};
        alg_batch.set_bfe(bfe{serial_bfe{}});
        pop_batch = alg_batch.evolve(pop_batch);

        BOOST_CHECK(pop_scalar.get_x() == pop_batch.get_x());
        BOOST_CHECK(pop_scalar.get_f() == pop_batch.get_f());
        // and the same number of function evaluations
        BOOST_CHECK_EQUAL(pop_batch.get_problem().get_fevals() - batch_fevals0, scalar_fevals);

        // The stock default_bfe leads to the same population too
        population pop_default{udp, 52u, 23u};
        nsga3 alg_default{5u, 1.00, 30., 0.10, 20., 5u, 0u, random_mating, 42u, false};
        alg_default.set_bfe(bfe{default_bfe{}});
        pop_default = alg_default.evolve(pop_default);
        BOOST_CHECK(pop_scalar.get_x() == pop_default.get_x());
        BOOST_CHECK(pop_scalar.get_f() == pop_default.get_f());
    }
}

BOOST_AUTO_TEST_CASE(nsga3_bfe_serialization)
{
    // A configured bfe survives a serialization round trip and keeps evolving identically
    dtlz udp{2u, 10u, 3u};
    population pop{udp, 52u, 23u};

    algorithm algo{nsga3{2u, 1.00, 30., 0.10, 20., 5u, 0u, true, 42u, false}};
    algo.extract<nsga3>()->set_bfe(bfe{serial_bfe{}});
    pop = algo.evolve(pop);

    std::stringstream ss;
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << algo;
    }
    algorithm restored{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> restored;
    }

    population continued_direct{pop};
    population continued_restored{pop};
    continued_direct = algo.evolve(continued_direct);
    continued_restored = restored.evolve(continued_restored);
    BOOST_CHECK(continued_direct.get_x() == continued_restored.get_x());
    BOOST_CHECK(continued_direct.get_f() == continued_restored.get_f());
}

BOOST_AUTO_TEST_CASE(nsga3_generation_equivalence)
{
    /*  The permutations driving the mating are rebuilt from the identity at the
     *  start of every generation, so a single evolution of N generations and N
     *  successive evolutions of one generation are the same computation. This holds
     *  with the inter-generational memory both enabled and disabled.
     */
    dtlz udp{2u, 10u, 3u};

    for (bool use_memory : {false, true}) {
        for (bool random_mating : {true, false}) {
            population pop_once{udp, 52u, 23u};
            nsga3 alg_once{5u, 1.00, 30., 0.10, 20., 5u, 0u, random_mating, 42u, use_memory};
            pop_once = alg_once.evolve(pop_once);

            population pop_stepped{udp, 52u, 23u};
            nsga3 alg_stepped{1u, 1.00, 30., 0.10, 20., 5u, 0u, random_mating, 42u, use_memory};
            for (unsigned i = 0u; i < 5u; ++i) {
                pop_stepped = alg_stepped.evolve(pop_stepped);
            }

            BOOST_CHECK(pop_once.get_x() == pop_stepped.get_x());
            BOOST_CHECK(pop_once.get_f() == pop_stepped.get_f());
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga3_memory_changes_the_evolution)
{
    /*  The running ideal point and the retained extreme points of Section IV-C are
     *  a different normalization from the per-generation one of Algorithm 2, and
     *  the flag must actually reach the pipeline.
     *
     *  Five objectives are used deliberately. While the first non-dominated front
     *  still fits in the population it is preserved whole by elitism, and the
     *  individual attaining the best value of an objective, or minimising an
     *  achievement scalarization, therefore survives; the retained quantities then
     *  agree with the recomputed ones and the two settings coincide. Once the first
     *  front is larger than the population, niching starts discarding such
     *  individuals and the two part company.
     */
    dtlz udp{2u, 10u, 5u};

    population pop_plain{udp, 52u, 23u};
    nsga3 alg_plain{30u, 1.00, 30., 0.10, 20., 3u, 0u, true, 42u, false};
    pop_plain = alg_plain.evolve(pop_plain);

    population pop_memory{udp, 52u, 23u};
    nsga3 alg_memory{30u, 1.00, 30., 0.10, 20., 3u, 0u, true, 42u, true};
    pop_memory = alg_memory.evolve(pop_memory);

    BOOST_CHECK(pop_plain.get_f() != pop_memory.get_f());
    // Both remain well formed
    for (const auto &f : pop_memory.get_f()) {
        BOOST_REQUIRE_EQUAL(f.size(), 5u);
        for (double value : f) {
            BOOST_CHECK(std::isfinite(value));
        }
    }

    // A memory-enabled run is reproducible in its own right
    population pop_memory_again{udp, 52u, 23u};
    nsga3 alg_memory_again{30u, 1.00, 30., 0.10, 20., 3u, 0u, true, 42u, true};
    pop_memory_again = alg_memory_again.evolve(pop_memory_again);
    BOOST_CHECK(pop_memory.get_x() == pop_memory_again.get_x());
    BOOST_CHECK(pop_memory.get_f() == pop_memory_again.get_f());
}

BOOST_AUTO_TEST_CASE(nsga3_instance_independence)
{
    dtlz udp{1u, 10u, 3u};

    // Baseline run
    population pop_ref{udp, 52u, 23u};
    nsga3 alg_ref{5u, 1.00, 30., 0.10, 20., 5u, 0u, true, 42u, false};
    pop_ref = alg_ref.evolve(pop_ref);

    /*  A differently seeded instance evolved in between, and a reseeded global
     *  random device, must leave an identically seeded run unchanged.
     */
    population pop_other{udp, 52u, 23u};
    nsga3 alg_other{5u, 1.00, 30., 0.10, 20., 5u, 0u, true, 7u, false};
    pop_other = alg_other.evolve(pop_other);
    random_device::set_seed(987654u);

    population pop_test{udp, 52u, 23u};
    nsga3 alg_test{5u, 1.00, 30., 0.10, 20., 5u, 0u, true, 42u, false};
    pop_test = alg_test.evolve(pop_test);

    BOOST_CHECK(pop_ref.get_x() == pop_test.get_x());
    BOOST_CHECK(pop_ref.get_f() == pop_test.get_f());
    // A differently seeded instance really does explore differently
    BOOST_CHECK(pop_ref.get_f() != pop_other.get_f());

    // Constructing an nsga3 must not disturb the global random device
    random_device::set_seed(4242u);
    const unsigned expected = random_device::next();
    random_device::set_seed(4242u);
    nsga3 constructed{5u, 1.00, 30., 0.10, 20., 5u, 0u, true, 32u, false};
    BOOST_CHECK_EQUAL(constructed.get_seed(), 32u);
    BOOST_CHECK_EQUAL(expected, random_device::next());

    // set_seed reseeds the engine, and is what the seed getter reports
    nsga3 reseeded{5u, 1.00, 30., 0.10, 20., 5u, 0u, true, 7u, false};
    reseeded.set_seed(42u);
    BOOST_CHECK_EQUAL(reseeded.get_seed(), 42u);
    population pop_reseeded{udp, 52u, 23u};
    pop_reseeded = reseeded.evolve(pop_reseeded);
    BOOST_CHECK(pop_ref.get_f() == pop_reseeded.get_f());
}

BOOST_AUTO_TEST_CASE(nsga3_log_generation_numbers)
{
    dtlz udp{1u, 10u, 3u};

    // Verbosity 1 logs every generation, numbered from 1
    population pop1{udp, 52u, 23u};
    nsga3 alg1{5u, 1.00, 30., 0.10, 20., 5u, 0u, true, 32u, false};
    alg1.set_verbosity(1u);
    pop1 = alg1.evolve(pop1);
    const auto &log1 = alg1.get_log();
    BOOST_REQUIRE_EQUAL(log1.size(), 5u);
    for (unsigned i = 0u; i < log1.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(log1[i]), i + 1u);
        BOOST_CHECK_EQUAL(std::get<2>(log1[i]).size(), udp.get_nobj());
    }
    // Function evaluations accumulate across generations
    for (std::size_t i = 1u; i < log1.size(); ++i) {
        BOOST_CHECK(std::get<1>(log1[i]) > std::get<1>(log1[i - 1u]));
    }

    // Verbosity 2 logs generations 1, 3 and 5
    population pop2{udp, 52u, 23u};
    nsga3 alg2{5u, 1.00, 30., 0.10, 20., 5u, 0u, true, 32u, false};
    alg2.set_verbosity(2u);
    pop2 = alg2.evolve(pop2);
    const auto &log2 = alg2.get_log();
    BOOST_REQUIRE_EQUAL(log2.size(), 3u);
    BOOST_CHECK_EQUAL(std::get<0>(log2[0]), 1u);
    BOOST_CHECK_EQUAL(std::get<0>(log2[1]), 3u);
    BOOST_CHECK_EQUAL(std::get<0>(log2[2]), 5u);

    // Verbosity 0 logs nothing
    population pop3{udp, 52u, 23u};
    nsga3 alg3{5u, 1.00, 30., 0.10, 20., 5u, 0u, true, 32u, false};
    pop3 = alg3.evolve(pop3);
    BOOST_CHECK(alg3.get_log().empty());

    // The log is cleared at the start of each evolution
    pop2 = alg2.evolve(pop2);
    BOOST_CHECK_EQUAL(alg2.get_log().size(), 3u);
}

static void nsga3_verify_serialization_continuation(bool use_memory)
{
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};

    algorithm algo{nsga3{3u, 1.00, 30., 0.10, 20., 5u, 0u, true, 32u, use_memory}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    /*  Round-trip the *evolved* algorithm. Continuing the evolution from the
     *  restored copy requires the engine state, the inter-generational memory
     *  and the constructor arguments to have all been archived.
     */
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << algo;
    }
    algorithm restored{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> restored;
    }

    BOOST_CHECK_EQUAL(algo.get_extra_info(), restored.get_extra_info());
    BOOST_CHECK(algo.extract<nsga3>()->get_log() == restored.extract<nsga3>()->get_log());

    population continued_direct{pop};
    population continued_restored{pop};
    continued_direct = algo.evolve(continued_direct);
    continued_restored = restored.evolve(continued_restored);

    BOOST_CHECK(continued_direct.get_x() == continued_restored.get_x());
    BOOST_CHECK(continued_direct.get_f() == continued_restored.get_f());
    BOOST_CHECK(algo.extract<nsga3>()->get_log() == restored.extract<nsga3>()->get_log());
}

BOOST_AUTO_TEST_CASE(nsga3_serialization_continuation)
{
    nsga3_verify_serialization_continuation(false);
    nsga3_verify_serialization_continuation(true);
}

BOOST_AUTO_TEST_CASE(nsga3_serialization_test)
{
    const double close_distance = 1e-8;
    problem prob{zdt{1u, 30u}};
    population pop{prob, 40u, 23u};
    algorithm algo{nsga3{10u, 1.00, 30., 0.10, 20, 5u, 0u, true, 32u, false}};
    algo.set_verbosity(1u);
    algo.set_seed(1234u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<nsga3>()->get_log();
    // Now serialize, deserialize and compare the result.
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << algo;
    }
    // Reset the algorithm instance before deserialization
    algo = algorithm{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> algo;
    }
    auto after_text = boost::lexical_cast<std::string>(algo);
    auto after_log = algo.extract<nsga3>()->get_log();

    BOOST_CHECK_EQUAL(before_text, after_text);
    BOOST_CHECK(before_log == after_log);
    BOOST_CHECK(before_log.size() > 0u);

    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_EQUAL(std::get<1>(before_log[i]), std::get<1>(after_log[i]));
        for (auto j = 0u; j < 2u; ++j) {
            BOOST_CHECK_CLOSE(std::get<2>(before_log[i])[j], std::get<2>(after_log[i])[j], close_distance);
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga3_zdt5_test)
{
    algorithm algo{nsga3(100u, 1.00, 30., 0.10, 20., 4u, 0u, true, 32u, false)};
    algo.set_verbosity(10u);
    algo.set_seed(23456u);
    population pop{zdt(5u, 10u), 20u, 32u};
    pop = algo.evolve(pop);

    // The integer chromosome of zdt5 survives crossover and mutation intact
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        auto x = pop.get_x()[i];
        BOOST_CHECK(std::all_of(x.begin(), x.end(), [](double el) { return (el == std::floor(el)); }));
    }

    // and the evolution is otherwise well formed on a discrete problem too
    BOOST_CHECK_EQUAL(pop.size(), 20u);
    const auto objs = pop.get_f();
    for (const auto &f : objs) {
        BOOST_REQUIRE_EQUAL(f.size(), 2u);
        for (double value : f) {
            BOOST_CHECK(std::isfinite(value));
        }
    }
    // After 100 generations a good share of the population is nondominated
    auto fronts = std::get<0>(fast_non_dominated_sorting(objs));
    BOOST_CHECK(fronts[0].size() * 2u >= pop.size());
}
