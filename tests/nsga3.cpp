#define BOOST_TEST_MODULE nsga3_test
#define BOOST_TEST_DYN_LINK
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga3.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/detail/nsga3_impl.hpp>
#include <pagmo/detail/reference_point.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

using namespace pagmo;


BOOST_AUTO_TEST_CASE(nsga3_algorithm_construction)
{
    BOOST_CHECK_NO_THROW(nsga3{});
    nsga3 user_algo{1u, 1.00, 30.0, 0.10, 20.0, 12u, 32u, false};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 32u);
    BOOST_CHECK(user_algo.get_extra_info().find("Seed: 32") != std::string::npos);

    // Verify throw on invalid arguments
    // Invalid cr
    BOOST_CHECK_THROW((nsga3{1u, 2.00, 30.0, 0.10, 20.0, 12u, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, -1.00, 30.0, 0.10, 20.0, 12u, 32u, false}), std::invalid_argument);
    // Invalid mut
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 1.10, 20.0, 12u, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, -0.10, 20.0, 12u, 32u, false}), std::invalid_argument);
    // Invalid eta_mut
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 0.10, 100.1, 12u, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 0.10, -0.1, 12u, 32u, false}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_evolve_population){
    dtlz udp{1u, 10u, 3u};
    problem prob{udp};
    const auto bounds = prob.get_bounds();

    population pop1{prob, 92u, 23u /*seed*/};
    const auto fevals0 = pop1.get_problem().get_fevals();

    nsga3 user_algo1{10u, 1.0, 30., 0.10, 20., 12u, 32u, false};
    BOOST_CHECK(user_algo1.get_seed() == 32u);
    user_algo1.set_verbosity(10u);
    pop1 = user_algo1.evolve(pop1);

    // The population size is preserved across the evolution
    BOOST_CHECK_EQUAL(pop1.size(), 92u);
    // Every individual is a usable, in-bounds solution with a finite fitness
    for(const auto &x: pop1.get_x()){
        BOOST_REQUIRE_EQUAL(x.size(), bounds.first.size());
        for(size_t i=0; i<x.size(); i++){
            BOOST_CHECK(x[i] >= bounds.first[i]);
            BOOST_CHECK(x[i] <= bounds.second[i]);
        }
    }
    for(const auto &f: pop1.get_f()){
        BOOST_REQUIRE_EQUAL(f.size(), prob.get_nobj());
        for(double value: f){
            BOOST_CHECK(std::isfinite(value));
        }
    }
    // Each generation evaluates a full offspring population
    BOOST_CHECK_EQUAL(pop1.get_problem().get_fevals() - fevals0, 10u*92u);
};

BOOST_AUTO_TEST_CASE(nsga3_reference_point_type){
    detail::reference_point rp3(3);
    BOOST_CHECK_EQUAL(rp3.dim(), 3);
    BOOST_CHECK_EQUAL(rp3[0], 0.0);
    BOOST_CHECK_EQUAL(rp3[1], 0.0);
    BOOST_CHECK_EQUAL(rp3[2], 0.0);
}

BOOST_AUTO_TEST_CASE(nsga3_verify_uniform_reference_points){
    /*  1. Verify cardinality of ref point set
     *  2. Verify coefficients sum to 1.0
     */

    double close_distance = 1e-8;
    auto rp_3_12 = detail::generate_uniform_reference_points(3, 12);
    BOOST_CHECK_EQUAL(rp_3_12.size(), 91);
    for(auto& p: rp_3_12){
        double p_sum = 0.0;
        for(size_t idx=0; idx<p.dim(); idx++){
            p_sum += p[idx];
        }
        BOOST_CHECK_CLOSE(p_sum, 1.0, 1e-8);
    }

    auto rp_8_12 = detail::generate_uniform_reference_points(8, 12);
    BOOST_CHECK_EQUAL(rp_8_12.size(), 50388);
    for(auto& p: rp_8_12){
        double p_sum = 0.0;
        for(size_t idx=0; idx<p.dim(); idx++){
            p_sum += p[idx];
        }
        BOOST_CHECK_CLOSE(p_sum, 1.0, close_distance);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_reference_points_exact){
    /*  The Das and Dennis recursion is easy to get subtly wrong, so pin the exact
     *  point sets, in generation order, for the smallest cases.
     */
    auto rp_2_2 = detail::generate_uniform_reference_points(2, 2);
    const std::vector<std::vector<double>> expected_2_2{{0.0, 1.0}, {0.5, 0.5}, {1.0, 0.0}};
    BOOST_REQUIRE_EQUAL(rp_2_2.size(), expected_2_2.size());
    for(size_t i=0; i<expected_2_2.size(); i++){
        BOOST_REQUIRE_EQUAL(rp_2_2[i].dim(), expected_2_2[i].size());
        for(size_t j=0; j<expected_2_2[i].size(); j++){
            BOOST_CHECK_SMALL(rp_2_2[i][j] - expected_2_2[i][j], 1e-12);
        }
    }

    // One division per objective yields exactly the canonical axis directions
    auto rp_3_1 = detail::generate_uniform_reference_points(3, 1);
    const std::vector<std::vector<double>> expected_3_1{{0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}};
    BOOST_REQUIRE_EQUAL(rp_3_1.size(), expected_3_1.size());
    for(size_t i=0; i<expected_3_1.size(); i++){
        for(size_t j=0; j<expected_3_1[i].size(); j++){
            BOOST_CHECK_SMALL(rp_3_1[i][j] - expected_3_1[i][j], 1e-12);
        }
    }

    auto rp_3_2 = detail::generate_uniform_reference_points(3, 2);
    const std::vector<std::vector<double>> expected_3_2{{0.0, 0.0, 1.0}, {0.0, 0.5, 0.5}, {0.0, 1.0, 0.0},
                                                        {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}, {1.0, 0.0, 0.0}};
    BOOST_REQUIRE_EQUAL(rp_3_2.size(), expected_3_2.size());
    for(size_t i=0; i<expected_3_2.size(); i++){
        for(size_t j=0; j<expected_3_2[i].size(); j++){
            BOOST_CHECK_SMALL(rp_3_2[i][j] - expected_3_2[i][j], 1e-12);
        }
    }

    // Every generated point is on the unit simplex: non-negative and summing to one
    for(const auto &set: {rp_2_2, rp_3_1, rp_3_2}){
        for(const auto &p: set){
            double sum = 0.0;
            for(double c: p.get_coeffs()){
                BOOST_CHECK(c >= 0.0);
                sum += c;
            }
            BOOST_CHECK_CLOSE(sum, 1.0, 1e-8);
        }
    }

    // n_choose_k against hand values
    BOOST_CHECK_EQUAL(detail::n_choose_k(3, 2), 3u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(4, 2), 6u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(5, 3), 10u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(7, 5), 21u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(14, 12), 91u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(10, 3), 120u);
    BOOST_CHECK_EQUAL(detail::n_choose_k(9, 0), 1u);

    /*  evolve() sizes the population against n_choose_k(m + p - 1, p), so that
     *  count must agree with the number of points actually generated.
     */
    const std::vector<std::pair<size_t, size_t>> cases{{2, 1}, {2, 2}, {3, 1}, {3, 2}, {3, 3}, {5, 4}, {8, 2}};
    for(const auto &c: cases){
        size_t nobjs = c.first, divisions = c.second;
        BOOST_CHECK_EQUAL(detail::generate_uniform_reference_points(nobjs, divisions).size(),
                          detail::n_choose_k(nobjs + divisions - 1, divisions));
    }
}

BOOST_AUTO_TEST_CASE(nsga3_perpendicular_distance){
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

BOOST_AUTO_TEST_CASE(nsga3_achievement_scalarization){
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

BOOST_AUTO_TEST_CASE(nsga3_reference_point_candidates){
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
    BOOST_CHECK_EQUAL(random_a.value(), random_b.value());  // same seed, same choice
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

BOOST_AUTO_TEST_CASE(nsga3_association_golden){
    /*  Two objectives and two divisions give the three reference directions
     *  (0,1), (1/2,1/2) and (1,0). Each normalized point below has a hand computed
     *  nearest direction and distance.
     */
    auto rps = detail::generate_uniform_reference_points(2, 2);
    BOOST_REQUIRE_EQUAL(rps.size(), 3u);

    const std::vector<std::vector<double>> norm_objs{{0.0, 1.0}, {1.0, 0.0}, {0.5, 0.5}, {0.9, 0.1}};
    const std::vector<size_t> expected_nearest{0u, 2u, 1u, 2u};
    const std::vector<double> expected_distance{0.0, 0.0, 0.0, 0.1};

    for(size_t i=0; i<norm_objs.size(); i++){
        size_t nearest = 0;
        double min_dist = std::numeric_limits<double>::max();
        for(size_t p=0; p<rps.size(); p++){
            double dist = detail::perpendicular_distance(rps[p].get_coeffs(), norm_objs[i]);
            if(dist < min_dist){
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

    BOOST_CHECK_EQUAL(rps[0].member_count(), 1u);   // individual 0
    BOOST_CHECK_EQUAL(rps[1].member_count(), 0u);
    BOOST_CHECK_EQUAL(rps[2].member_count(), 1u);   // individual 1
    BOOST_CHECK_EQUAL(rps[0].candidate_count(), 0u);
    BOOST_CHECK_EQUAL(rps[1].candidate_count(), 1u);  // individual 2
    BOOST_CHECK_EQUAL(rps[2].candidate_count(), 1u);  // individual 3
    BOOST_REQUIRE(rps[1].nearest_candidate().has_value());
    BOOST_CHECK_EQUAL(rps[1].nearest_candidate().value(), 2u);
    BOOST_REQUIRE(rps[2].nearest_candidate().has_value());
    BOOST_CHECK_EQUAL(rps[2].nearest_candidate().value(), 3u);

    /*  A single front is the last front, so every individual becomes a candidate
     *  and none becomes a member.
     */
    auto single = detail::generate_uniform_reference_points(2, 2);
    const std::vector<std::vector<pop_size_t>> one_front{{0u, 1u, 2u, 3u}};
    detail::associate_with_reference_points(single, norm_objs, one_front);
    size_t total_members = 0, total_candidates = 0;
    for(const auto &rp: single){
        total_members += rp.member_count();
        total_candidates += rp.candidate_count();
    }
    BOOST_CHECK_EQUAL(total_members, 0u);
    BOOST_CHECK_EQUAL(total_candidates, norm_objs.size());
}

BOOST_AUTO_TEST_CASE(nsga3_niching_tie_breaking){
    detail::random_engine_type reng(1234u);

    // A unique least crowded reference point is returned without consulting the engine
    auto rps = detail::generate_uniform_reference_points(2, 2);
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
    auto tied = detail::generate_uniform_reference_points(3, 2);
    BOOST_REQUIRE_EQUAL(tied.size(), 6u);
    detail::random_engine_type reng_a(99u), reng_b(99u);
    size_t pick_a = detail::identify_niche_point(tied, reng_a);
    size_t pick_b = detail::identify_niche_point(tied, reng_b);
    BOOST_CHECK_EQUAL(pick_a, pick_b);
    BOOST_CHECK(pick_a < tied.size());

    // Only reference points at the minimum member count are eligible
    for(size_t i=0; i<tied.size(); i++){
        if(i != 2u){
            tied[i].increment_members();
        }
    }
    BOOST_CHECK_EQUAL(detail::identify_niche_point(tied, reng), 2u);
}

BOOST_AUTO_TEST_CASE(nsga3_selection_golden){
    /*  Golden environmental selection over fixed objective vectors. The fixtures
     *  are chosen so that the outcome does not depend on how ties are broken:
     *  std::uniform_int_distribution is not specified to map engine output to
     *  values identically across standard library implementations, so an index
     *  drawn from a set of two or more cannot be asserted portably.
     */
    detail::random_engine_type reng(17u);

    /*  Fixture A: indices 0-3 are the first front, 4 is the whole second front and
     *  5 is dominated by 4. One slot is left after the first front and exactly one
     *  candidate can fill it, so the result is completely determined.
     */
    const std::vector<vector_double> objs_a{{0.0, 1.0}, {0.1, 0.9}, {0.5, 0.5}, {0.45, 0.55},
                                            {0.6, 0.6}, {2.0, 0.6}};
    auto fronts_a = std::get<0>(fast_non_dominated_sorting(objs_a));
    BOOST_REQUIRE_EQUAL(fronts_a.size(), 3u);
    BOOST_REQUIRE_EQUAL(fronts_a[0].size(), 4u);

    auto next_a = detail::nsga3_selection(objs_a, 5u, 2u, nullptr, nullptr, reng);
    const std::vector<size_t> expected_a{0u, 1u, 2u, 3u, 4u};
    BOOST_CHECK(next_a == expected_a);

    /*  Fixture B: the second front is absorbed whole, so every individual survives
     *  whatever order the niching visits the reference points in.
     */
    const std::vector<vector_double> objs_b{{0.0, 1.0}, {0.1, 0.9}, {0.5, 0.5}, {1.0, 0.0},
                                            {0.6, 0.6}, {1.1, 0.1}};
    auto next_b = detail::nsga3_selection(objs_b, 6u, 2u, nullptr, nullptr, reng);
    std::vector<size_t> sorted_b{next_b};
    std::sort(sorted_b.begin(), sorted_b.end());
    const std::vector<size_t> expected_b{0u, 1u, 2u, 3u, 4u, 5u};
    BOOST_CHECK(sorted_b == expected_b);

    /*  Fixture C: two candidates compete for one slot, so which one survives is a
     *  random tie-break. The structural guarantees still hold exactly, and the
     *  choice must be reproducible for a given engine state.
     */
    detail::random_engine_type reng_c1(5u), reng_c2(5u);
    auto next_c1 = detail::nsga3_selection(objs_b, 5u, 2u, nullptr, nullptr, reng_c1);
    auto next_c2 = detail::nsga3_selection(objs_b, 5u, 2u, nullptr, nullptr, reng_c2);
    BOOST_CHECK(next_c1 == next_c2);  // same seed, same survivors

    BOOST_CHECK_EQUAL(next_c1.size(), 5u);
    std::vector<size_t> sorted_c{next_c1};
    std::sort(sorted_c.begin(), sorted_c.end());
    BOOST_CHECK(std::unique(sorted_c.begin(), sorted_c.end()) == sorted_c.end());  // no duplicates
    for(size_t idx: next_c1){
        BOOST_CHECK(idx < objs_b.size());
    }
    // The whole first front survives, and the last slot comes from the second front
    auto fronts_b = std::get<0>(fast_non_dominated_sorting(objs_b));
    BOOST_REQUIRE_EQUAL(fronts_b[0].size(), 4u);
    for(auto idx: fronts_b[0]){
        BOOST_CHECK(std::find(next_c1.begin(), next_c1.end(), idx) != next_c1.end());
    }
    size_t from_last_front = 0;
    for(auto idx: fronts_b[1]){
        from_last_front += (std::find(next_c1.begin(), next_c1.end(), idx) != next_c1.end()) ? 1u : 0u;
    }
    BOOST_CHECK_EQUAL(from_last_front, 1u);
}

BOOST_AUTO_TEST_CASE(nsga3_selection_preserves_size){
    /*  Environmental selection over a larger, structured set: whatever the fronts
     *  look like, exactly N_pop distinct in-range indices come back.
     */
    dtlz udp{2u, 10u, 3u};
    population pop{udp, 64u, 23u};
    detail::random_engine_type reng(3u);

    auto objs = pop.get_f();
    for(size_t n_pop: {8u, 16u, 32u, 63u}){
        auto next = detail::nsga3_selection(objs, n_pop, 4u, nullptr, nullptr, reng);
        BOOST_CHECK_EQUAL(next.size(), n_pop);
        std::vector<size_t> sorted_next{next};
        std::sort(sorted_next.begin(), sorted_next.end());
        BOOST_CHECK(std::unique(sorted_next.begin(), sorted_next.end()) == sorted_next.end());
        for(size_t idx: next){
            BOOST_CHECK(idx < objs.size());
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_translate_objectives){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};

    pop = nsga3_alg.evolve(pop);
    auto p0_obj = pop.get_f();
    auto ideal_point = detail::nsga3_compute_ideal(p0_obj, nullptr);
    auto translated_objectives = detail::nsga3_translate_objectives(p0_obj, ideal_point);

    BOOST_REQUIRE_EQUAL(translated_objectives.size(), p0_obj.size());
    BOOST_REQUIRE_EQUAL(ideal_point.size(), udp.get_nobj());

    // The ideal point is the componentwise minimum of the objectives
    for(size_t obj=0; obj<ideal_point.size(); obj++){
        double column_min = std::numeric_limits<double>::max();
        for(const auto &f: p0_obj){
            column_min = std::min(column_min, f[obj]);
        }
        BOOST_CHECK_EQUAL(ideal_point[obj], column_min);
    }

    // Translation is an exact componentwise shift by that point
    for(size_t i=0; i<p0_obj.size(); i++){
        BOOST_REQUIRE_EQUAL(translated_objectives[i].size(), ideal_point.size());
        for(size_t obj=0; obj<ideal_point.size(); obj++){
            BOOST_CHECK_EQUAL(translated_objectives[i][obj], p0_obj[i][obj] - ideal_point[obj]);
            // Which leaves the whole population in the non-negative orthant
            BOOST_CHECK(translated_objectives[i][obj] >= 0.0);
        }
    }

    // and puts the origin at the ideal point: every objective attains zero
    for(size_t obj=0; obj<ideal_point.size(); obj++){
        double column_min = std::numeric_limits<double>::max();
        for(const auto &row: translated_objectives){
            column_min = std::min(column_min, row[obj]);
        }
        BOOST_CHECK_SMALL(column_min, 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_gaussian_elimination){
    // Verify correctness of simple system
    std::vector<std::vector<double>> A(3);
    std::vector<double> b = {1.0, 1.0, 1.0};

    A[0] = {-1, 1,  2};
    A[1] = { 2, 0, -3};
    A[2] = { 5, 1, -2};

    auto x = detail::gaussian_elimination(A, b);
    BOOST_REQUIRE(x.has_value());
    BOOST_CHECK_CLOSE((*x)[0], -0.4, 1e-8);
    BOOST_CHECK_CLOSE((*x)[1],  1.8, 1e-8);
    BOOST_CHECK_CLOSE((*x)[2], -0.6, 1e-8);

    /*  A zero leading pivot is not an error: partial pivoting selects the largest
     *  available pivot in each column, so this non-singular system is solvable.
     */
    std::vector<std::vector<double>> pivoted{{0.0, 2.0, 1.0}, {1.0, 0.0, 3.0}, {2.0, 1.0, 0.0}};
    auto xp = detail::gaussian_elimination(pivoted, b);
    BOOST_REQUIRE(xp.has_value());
    for(size_t i=0; i<pivoted.size(); i++){
        double residual = 0.0;
        for(size_t j=0; j<pivoted[i].size(); j++){
            residual += pivoted[i][j]*(*xp)[j];
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

    // Dimensions are validated
    std::vector<std::vector<double>> empty_matrix;
    BOOST_CHECK_THROW((detail::gaussian_elimination(empty_matrix, b)), std::invalid_argument);
    std::vector<std::vector<double>> non_square{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    BOOST_CHECK_THROW((detail::gaussian_elimination(non_square, b)), std::invalid_argument);
    std::vector<double> short_b{1.0, 1.0};
    BOOST_CHECK_THROW((detail::gaussian_elimination(A, short_b)), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_test_extreme_point_duplicates){
    // Translated objectives whose nadir point, (1, 1, 1), differs from the solved intercepts
    const std::vector<std::vector<double>> translated{{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}};

    /*  These extreme points are pairwise distinct, but every pair shares at least
     *  one coordinate. Comparing coordinates individually misclassifies them as
     *  duplicates and skips the solver; comparing complete vectors does not.
     */
    std::vector<std::vector<double>> distinct{{2.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 8.0}};
    auto intercepts = detail::nsga3_find_intercepts(distinct, translated);
    BOOST_CHECK_CLOSE(intercepts[0], 2.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[2], 8.0, 1e-8);

    // Two identical extreme points do fall back to the nadir point
    std::vector<std::vector<double>> duplicated{{2.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 0.0, 8.0}};
    intercepts = detail::nsga3_find_intercepts(duplicated, translated);
    BOOST_CHECK_CLOSE(intercepts[0], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[2], 1.0, 1e-8);

    // As do points which differ only within the numerical tolerance
    std::vector<std::vector<double>> near_duplicated{{2.0, 0.0, 0.0}, {2.0 + 1e-15, 1e-16, -1e-16},
                                                     {0.0, 0.0, 8.0}};
    intercepts = detail::nsga3_find_intercepts(near_duplicated, translated);
    BOOST_CHECK_CLOSE(intercepts[0], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[2], 1.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(nsga3_test_intercepts_fallback){
    // Front 0 is {(1,3), (3,1)}; (4,4) is dominated, so the nadir point is (3, 3)
    const std::vector<std::vector<double>> translated{{1.0, 3.0}, {3.0, 1.0}, {4.0, 4.0}};

    // A negative solution component has no usable reciprocal
    std::vector<std::vector<double>> negative{{-1.0, 0.0}, {0.0, 2.0}};
    auto intercepts = detail::nsga3_find_intercepts(negative, translated);
    BOOST_CHECK_CLOSE(intercepts[0], 3.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 3.0, 1e-8);

    // A singular system falls back rather than aborting the evolution
    std::vector<std::vector<double>> singular{{0.0, 0.0}, {0.0, 2.0}};
    BOOST_CHECK_NO_THROW((intercepts = detail::nsga3_find_intercepts(singular, translated)));
    BOOST_CHECK_CLOSE(intercepts[0], 3.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 3.0, 1e-8);

    /*  A degenerate objective, identical across the population, has zero extent.
     *  The intercept is sanitised to 1.0 so that normalization leaves the
     *  coordinate at zero instead of producing an infinity or a NaN.
     */
    const std::vector<std::vector<double>> degenerate{{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}};
    std::vector<std::vector<double>> degenerate_ext{{0.0, 1.0}, {0.0, 1.0}};
    auto degenerate_intercepts = detail::nsga3_find_intercepts(degenerate_ext, degenerate);
    BOOST_CHECK_CLOSE(degenerate_intercepts[0], 1.0, 1e-8);
    auto norm_objs = detail::nsga3_normalize_objectives(degenerate, degenerate_intercepts);
    for(const auto &row: norm_objs){
        BOOST_CHECK_EQUAL(row[0], 0.0);
        for(double value: row){
            BOOST_CHECK(std::isfinite(value));
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_normalize_nonzero_ideal){
    /*  Ideal point (2, 5, -1), well away from the origin. The first three points
     *  are mutually non-dominated and the fourth is dominated by all of them.
     */
    const std::vector<vector_double> objs{{2.0, 9.0, 3.0}, {6.0, 5.0, 3.0}, {6.0, 9.0, -1.0}, {6.0, 9.0, 3.0}};

    auto ideal_point = detail::nsga3_compute_ideal(objs, nullptr);
    BOOST_CHECK_CLOSE(ideal_point[0],  2.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal_point[1],  5.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal_point[2], -1.0, 1e-8);

    auto translated = detail::nsga3_translate_objectives(objs, ideal_point);
    for(size_t obj=0; obj<ideal_point.size(); obj++){
        double col_min = std::numeric_limits<double>::max();
        for(const auto &row: translated){
            col_min = std::min(col_min, row[obj]);
        }
        BOOST_CHECK_SMALL(col_min, 1e-12);
    }

    const std::vector<std::vector<pop_size_t>> fronts{{0u, 1u, 2u}};
    auto ext_points = detail::nsga3_find_extreme_points(fronts, translated, ideal_point, nullptr);
    auto intercepts = detail::nsga3_find_intercepts(ext_points, translated);

    /*  The extreme points coincide here, so the nadir fallback is taken. The
     *  intercepts must be the nadir of the *translated* objectives, (4, 4, 4),
     *  and not the nadir of the original objectives, (6, 9, 3).
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
    auto spread_ideal = detail::nsga3_compute_ideal(spread_objs, nullptr);
    BOOST_CHECK_CLOSE(spread_ideal[0],  2.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_ideal[1],  5.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_ideal[2], -1.0, 1e-8);
    auto spread_translated = detail::nsga3_translate_objectives(spread_objs, spread_ideal);
    auto spread_ext = detail::nsga3_find_extreme_points(fronts, spread_translated, spread_ideal, nullptr);
    auto spread_intercepts = detail::nsga3_find_intercepts(spread_ext, spread_translated);
    BOOST_CHECK_CLOSE(spread_intercepts[0], 9.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_intercepts[1], 5.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_intercepts[2], 1.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(nsga3_test_memory_ideal_shift){
    const std::vector<std::vector<pop_size_t>> fronts{{0u, 1u, 2u}};

    // Generation 1: the ideal point is (10, 10, 10)
    const std::vector<vector_double> objs1{{19.0, 10.0, 10.0}, {10.0, 15.0, 10.0}, {10.0, 10.0, 11.0}};
    // Generation 2: the ideal point improves to (8, 9, 10)
    const std::vector<vector_double> objs2{{8.0, 30.0, 30.0}, {30.0, 9.0, 30.0}, {30.0, 30.0, 10.0}};

    /*  With memory enabled the running ideal point and the retained extreme
     *  points persist across generations; nsga3 owns these two buffers.
     */
    std::vector<double> running_ideal;
    std::vector<std::vector<double>> retained_extremes;

    auto ideal1 = detail::nsga3_compute_ideal(objs1, &running_ideal);
    BOOST_CHECK_CLOSE(ideal1[0], 10.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal1[1], 10.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal1[2], 10.0, 1e-8);
    auto translated1 = detail::nsga3_translate_objectives(objs1, ideal1);
    auto ext1 = detail::nsga3_find_extreme_points(fronts, translated1, ideal1, &retained_extremes);
    // Translated extremes of generation 1: (9,0,0), (0,5,0), (0,0,1)
    BOOST_CHECK_CLOSE(ext1[0][0], 9.0, 1e-8);
    BOOST_CHECK_CLOSE(ext1[1][1], 5.0, 1e-8);
    BOOST_CHECK_CLOSE(ext1[2][2], 1.0, 1e-8);

    // The running ideal point is the elementwise minimum over both generations
    auto ideal2 = detail::nsga3_compute_ideal(objs2, &running_ideal);
    BOOST_CHECK_CLOSE(ideal2[0],  8.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal2[1],  9.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal2[2], 10.0, 1e-8);

    auto translated2 = detail::nsga3_translate_objectives(objs2, ideal2);
    auto ext2 = detail::nsga3_find_extreme_points(fronts, translated2, ideal2, &retained_extremes);

    /*  Generation 2 has no candidate better than the retained extreme points, so
     *  those are returned, re-expressed in the *current* translated coordinates:
     *  (19,10,10) - (8,9,10) = (11,1,0), and so on. Had the extreme points been
     *  retained in the translated coordinates of generation 1 they would still
     *  read (9,0,0), (0,5,0), (0,0,1) here.
     */
    const std::vector<std::vector<double>> expected_ext2{{11.0, 1.0, 0.0}, {2.0, 6.0, 0.0}, {2.0, 1.0, 1.0}};
    for(size_t i=0; i<expected_ext2.size(); i++){
        for(size_t j=0; j<expected_ext2[i].size(); j++){
            BOOST_CHECK_SMALL(ext2[i][j] - expected_ext2[i][j], 1e-9);
        }
    }

    // A worse generation does not degrade the retained ideal point
    const std::vector<vector_double> objs3{{20.0, 20.0, 20.0}, {21.0, 21.0, 21.0}, {22.0, 22.0, 22.0}};
    auto ideal3 = detail::nsga3_compute_ideal(objs3, &running_ideal);
    BOOST_CHECK_CLOSE(ideal3[0],  8.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal3[1],  9.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal3[2], 10.0, 1e-8);

    /*  Without memory the same generation-2 input depends only on the current
     *  objectives: the extreme point for the first objective is (22, 0, 20).
     */
    auto plain_ideal = detail::nsga3_compute_ideal(objs2, nullptr);
    auto plain_translated = detail::nsga3_translate_objectives(objs2, plain_ideal);
    auto plain_ext = detail::nsga3_find_extreme_points(fronts, plain_translated, plain_ideal, nullptr);
    BOOST_CHECK_CLOSE(plain_ext[0][0], 22.0, 1e-8);
    BOOST_CHECK_SMALL(plain_ext[0][1], 1e-12);
    BOOST_CHECK_CLOSE(plain_ext[0][2], 20.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(nsga3_test_find_extreme_points){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};

    pop = nsga3_alg.evolve(pop);
    auto objs = pop.get_f();
    const size_t nobj = udp.get_nobj();
    auto ideal_point = detail::nsga3_compute_ideal(objs, nullptr);
    auto translated_objectives = detail::nsga3_translate_objectives(objs, ideal_point);
    auto fnds_res = fast_non_dominated_sorting(objs);
    auto fronts = std::get<0>(fnds_res);
    auto ext_points = detail::nsga3_find_extreme_points(fronts, translated_objectives, ideal_point, nullptr);

    BOOST_REQUIRE_EQUAL(ext_points.size(), nobj);

    for(size_t axis=0; axis<nobj; axis++){
        BOOST_REQUIRE_EQUAL(ext_points[axis].size(), nobj);

        /*  Recompute the achievement scalarization independently: the returned
         *  extreme point must be the individual of the first front that minimises
         *  it for this axis, which is the whole contract of the function.
         */
        std::vector<double> weights(nobj, 1e-6);
        weights[axis] = 1.0;
        double best_asf = std::numeric_limits<double>::max();
        for(auto idx: fronts[0]){
            best_asf = std::min(best_asf, detail::achievement(translated_objectives[idx], weights));
        }
        BOOST_CHECK_CLOSE(detail::achievement(ext_points[axis], weights), best_asf, 1e-8);

        // and it must be one of those individuals, not a synthesised point
        bool found = false;
        for(auto idx: fronts[0]){
            if(translated_objectives[idx] == ext_points[axis]){
                found = true;
                break;
            }
        }
        BOOST_CHECK(found);

        // Extremes come from the translated population, so they are non-negative
        for(double value: ext_points[axis]){
            BOOST_CHECK(std::isfinite(value));
            BOOST_CHECK(value >= 0.0);
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_find_intercepts){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    std::vector<double> intercepts{};

    pop = nsga3_alg.evolve(pop);
    auto objs = pop.get_f();
    auto ideal_point = detail::nsga3_compute_ideal(objs, nullptr);
    auto translated_objectives = detail::nsga3_translate_objectives(objs, ideal_point);
    auto fnds_res = fast_non_dominated_sorting(objs);
    auto fronts = std::get<0>(fnds_res);
    auto ext_points = detail::nsga3_find_extreme_points(fronts, translated_objectives, ideal_point, nullptr);

    intercepts = detail::nsga3_find_intercepts(ext_points, translated_objectives);
    BOOST_REQUIRE_EQUAL(intercepts.size(), udp.get_nobj());
    // Intercepts are always usable divisors
    for(double intercept: intercepts){
        BOOST_CHECK(std::isfinite(intercept));
        BOOST_CHECK(intercept > 0.0);
    }
    /*  Each extreme point, expressed in units of the intercepts, stays finite and
     *  positive. The stronger identity below cannot be asserted here because the
     *  nadir fallback legitimately breaks it whenever the extreme points coincide.
     */
    for(const auto &ext: ext_points){
        double plane = 0.0;
        for(size_t j=0; j<intercepts.size(); j++){
            plane += ext[j]/intercepts[j];
        }
        BOOST_CHECK(std::isfinite(plane));
        BOOST_CHECK(plane > 0.0);
    }

    /*  With well separated extreme points the solver path is taken, and the
     *  intercepts are by construction the axis crossings of the hyperplane through
     *  them: every extreme point then satisfies sum_j ext[j]/intercept[j] == 1.
     */
    const std::vector<std::vector<double>> spread_translated{{9.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 1.0},
                                                             {3.0, 2.0, 0.5}};
    const std::vector<std::vector<double>> spread_ext{{9.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 1.0}};
    auto spread_intercepts = detail::nsga3_find_intercepts(spread_ext, spread_translated);
    BOOST_CHECK_CLOSE(spread_intercepts[0], 9.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_intercepts[1], 5.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_intercepts[2], 1.0, 1e-8);
    for(const auto &ext: spread_ext){
        double plane = 0.0;
        for(size_t j=0; j<spread_intercepts.size(); j++){
            plane += ext[j]/spread_intercepts[j];
        }
        BOOST_CHECK_CLOSE(plane, 1.0, 1e-8);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_normalize_objectives){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    std::vector<std::vector<double>> norm_objs{};

    pop = nsga3_alg.evolve(pop);
    auto objs = pop.get_f();
    auto ideal_point = detail::nsga3_compute_ideal(objs, nullptr);
    auto translated_objectives = detail::nsga3_translate_objectives(objs, ideal_point);
    auto fnds_res = fast_non_dominated_sorting(objs);
    auto fronts = std::get<0>(fnds_res);
    auto ext_points = detail::nsga3_find_extreme_points(fronts, translated_objectives, ideal_point, nullptr);
    auto intercepts = detail::nsga3_find_intercepts(ext_points, translated_objectives);
    norm_objs = detail::nsga3_normalize_objectives(translated_objectives, intercepts);

    BOOST_REQUIRE_EQUAL(norm_objs.size(), translated_objectives.size());

    // Normalization divides each objective by its own intercept, and nothing else
    for(size_t i=0; i<translated_objectives.size(); i++){
        BOOST_REQUIRE_EQUAL(norm_objs[i].size(), intercepts.size());
        for(size_t j=0; j<intercepts.size(); j++){
            BOOST_CHECK_EQUAL(norm_objs[i][j], translated_objectives[i][j]/intercepts[j]);
            BOOST_CHECK(std::isfinite(norm_objs[i][j]));
            // The translated objectives are non-negative and the intercepts positive
            BOOST_CHECK(norm_objs[i][j] >= 0.0);
        }
    }

    // An individual sitting on the intercept vector lands on the unit hyperplane
    auto on_plane = detail::nsga3_normalize_objectives({intercepts}, intercepts);
    BOOST_REQUIRE_EQUAL(on_plane.size(), 1u);
    for(double value: on_plane[0]){
        BOOST_CHECK_CLOSE(value, 1.0, 1e-8);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_reproducibility_same_seed){
    dtlz udp{1u, 10u, 3u};

    population pop_a{udp, 52u, 23u};
    population pop_b{udp, 52u, 23u};
    nsga3 alg_a{5u, 1.00, 30., 0.10, 20., 5u, 42u, false};
    nsga3 alg_b{5u, 1.00, 30., 0.10, 20., 5u, 42u, false};
    alg_a.set_verbosity(1u);
    alg_b.set_verbosity(1u);

    pop_a = alg_a.evolve(pop_a);
    pop_b = alg_b.evolve(pop_b);

    BOOST_CHECK(pop_a.get_x() == pop_b.get_x());
    BOOST_CHECK(pop_a.get_f() == pop_b.get_f());
    BOOST_CHECK(alg_a.get_log() == alg_b.get_log());
}

BOOST_AUTO_TEST_CASE(nsga3_instance_independence){
    dtlz udp{1u, 10u, 3u};

    // Baseline run
    population pop_ref{udp, 52u, 23u};
    nsga3 alg_ref{5u, 1.00, 30., 0.10, 20., 5u, 42u, false};
    pop_ref = alg_ref.evolve(pop_ref);

    /*  A differently seeded instance evolved in between, and a reseeded global
     *  random device, must leave an identically seeded run unchanged.
     */
    population pop_other{udp, 52u, 23u};
    nsga3 alg_other{5u, 1.00, 30., 0.10, 20., 5u, 7u, false};
    pop_other = alg_other.evolve(pop_other);
    random_device::set_seed(987654u);

    population pop_test{udp, 52u, 23u};
    nsga3 alg_test{5u, 1.00, 30., 0.10, 20., 5u, 42u, false};
    pop_test = alg_test.evolve(pop_test);

    BOOST_CHECK(pop_ref.get_x() == pop_test.get_x());
    BOOST_CHECK(pop_ref.get_f() == pop_test.get_f());
    // A differently seeded instance really does explore differently
    BOOST_CHECK(pop_ref.get_f() != pop_other.get_f());

    // Constructing an nsga3 must not disturb the global random device
    random_device::set_seed(4242u);
    unsigned expected = random_device::next();
    random_device::set_seed(4242u);
    nsga3 constructed{5u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    BOOST_CHECK_EQUAL(constructed.get_seed(), 32u);
    BOOST_CHECK_EQUAL(expected, random_device::next());
}

BOOST_AUTO_TEST_CASE(nsga3_log_generation_numbers){
    dtlz udp{1u, 10u, 3u};

    // Verbosity 1 logs every generation, numbered from 1
    population pop1{udp, 52u, 23u};
    nsga3 alg1{5u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    alg1.set_verbosity(1u);
    pop1 = alg1.evolve(pop1);
    const auto &log1 = alg1.get_log();
    BOOST_REQUIRE_EQUAL(log1.size(), 5u);
    for(unsigned i=0; i<log1.size(); i++){
        BOOST_CHECK_EQUAL(std::get<0>(log1[i]), i + 1u);
    }
    // Function evaluations accumulate across generations
    for(size_t i=1; i<log1.size(); i++){
        BOOST_CHECK(std::get<1>(log1[i]) > std::get<1>(log1[i-1]));
    }

    // Verbosity 2 logs generations 1, 3 and 5
    population pop2{udp, 52u, 23u};
    nsga3 alg2{5u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    alg2.set_verbosity(2u);
    pop2 = alg2.evolve(pop2);
    const auto &log2 = alg2.get_log();
    BOOST_REQUIRE_EQUAL(log2.size(), 3u);
    BOOST_CHECK_EQUAL(std::get<0>(log2[0]), 1u);
    BOOST_CHECK_EQUAL(std::get<0>(log2[1]), 3u);
    BOOST_CHECK_EQUAL(std::get<0>(log2[2]), 5u);

    // Verbosity 0 logs nothing
    population pop3{udp, 52u, 23u};
    nsga3 alg3{5u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    pop3 = alg3.evolve(pop3);
    BOOST_CHECK(alg3.get_log().empty());
}

static void nsga3_verify_serialization_continuation(bool use_memory){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};

    algorithm algo{nsga3{3u, 1.00, 30., 0.10, 20., 5u, 32u, use_memory}};
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

BOOST_AUTO_TEST_CASE(nsga3_serialization_continuation){
    nsga3_verify_serialization_continuation(false);
    nsga3_verify_serialization_continuation(true);
}

BOOST_AUTO_TEST_CASE(nsga3_serialization_test){
    double close_distance = 1e-8;
    problem prob{zdt{1u, 30u}};
    population pop{prob, 40u, 23u};
    algorithm algo{nsga3{10u, 1.00, 30., 0.10, 20, 5u, 32u, false}};
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
    algorithm algo{nsga3(100u, 1.00, 30., 0.10, 20., 4u, 32u, false)};
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
