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

#define BOOST_TEST_MODULE mo_utilities_test

#include <boost/test/included/unit_test.hpp>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include <pagmo/io.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(pareto_dominance_test)
{
    BOOST_CHECK(pareto_dominance({1, 2, 3}, {4, 5, 6}));
    BOOST_CHECK(!pareto_dominance({4, 5, 6}, {4, 5, 6}));
    BOOST_CHECK(pareto_dominance({4, 5, 5}, {4, 5, 6}));
    BOOST_CHECK(!pareto_dominance({1, 2, 3}, {2, 1, 5}));
    BOOST_CHECK(pareto_dominance({-3.4, 1.5, 2.9, -2.3, 4.99, 3.2, 6.6}, {1, 2, 3, 4, 5, 6, 7}));
    BOOST_CHECK(!pareto_dominance({}, {}));
    BOOST_CHECK(pareto_dominance({2}, {3}));
    BOOST_CHECK_THROW(pareto_dominance({1, 2}, {3, 4, 5}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(non_dominated_front_2d_test)
{
    // Corner cases
    BOOST_CHECK(non_dominated_front_2d({}) == std::vector<vector_double::size_type>{});
    // We test some known cases
    {
        auto res = non_dominated_front_2d({{0, 1}, {1, 1}, {1, 2}});
        auto sol = std::vector<vector_double::size_type>{0u};
        BOOST_CHECK(std::is_permutation(res.begin(), res.end(), sol.begin()));
    }
    {
        auto res = non_dominated_front_2d({{0, 1}, {0, 1}, {-1, 2}, {-1, 2}});
        auto sol = std::vector<vector_double::size_type>{0u, 1u, 2u, 3u};
        BOOST_CHECK(std::is_permutation(res.begin(), res.end(), sol.begin()));
    }
    {
        auto res = non_dominated_front_2d({{0, 1}, {11, 9}, {6, 4}, {2, 4}, {4, 2}, {1, 0}});
        auto sol = std::vector<vector_double::size_type>{0u, 5u};
        BOOST_CHECK(std::is_permutation(res.begin(), res.end(), sol.begin()));
    }
    // And we test the throws
    BOOST_CHECK_THROW(non_dominated_front_2d({{1, 2}, {2}, {2, 3, 4}}), std::invalid_argument);
    BOOST_CHECK_THROW(non_dominated_front_2d({{2, 3, 2}, {1, 2, 5}, {2, 3, 4}}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(fast_non_dominated_sorting_test)
{
    // We create various values to compute
    std::vector<vector_double::size_type> dom_count;
    std::vector<std::vector<vector_double::size_type>> dom_list;
    std::vector<vector_double::size_type> non_dom_rank;

    // And the results to check
    std::vector<vector_double> example;
    std::vector<std::vector<vector_double::size_type>> non_dom_fronts_res;
    std::vector<vector_double::size_type> dom_count_res;
    std::vector<std::vector<vector_double::size_type>> dom_list_res;
    std::vector<vector_double::size_type> non_dom_rank_res;

    // Test 1
    example = {{0, 7}, {1, 5}, {2, 3}, {4, 2}, {7, 1}, {10, 0}, {2, 6}, {4, 4}, {10, 2}, {6, 6}, {9, 5}};
    non_dom_fronts_res = {{0, 1, 2, 3, 4, 5}, {6, 7, 8}, {9, 10}};
    dom_count_res = {0, 0, 0, 0, 0, 0, 2, 2, 3, 5, 5};
    dom_list_res = {{}, {6, 9, 10}, {6, 7, 9, 10}, {7, 8, 9, 10}, {8, 10}, {8}, {9}, {9, 10}, {}, {}, {}};
    non_dom_rank_res = {0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2};

    auto retval = fast_non_dominated_sorting(example);
    BOOST_CHECK(std::get<0>(retval) == non_dom_fronts_res);
    BOOST_CHECK(std::get<1>(retval) == dom_list_res);
    BOOST_CHECK(std::get<2>(retval) == dom_count_res);
    BOOST_CHECK(std::get<3>(retval) == non_dom_rank_res);

    // Test 2
    example = {{1, 2, 3}, {-2, 3, 7}, {-1, -2, -3}, {0, 0, 0}};
    non_dom_fronts_res = {{1, 2}, {3}, {0}};
    dom_count_res = {2, 0, 0, 1};
    dom_list_res = {{}, {}, {0, 3}, {0}};
    non_dom_rank_res = {2, 0, 0, 1};

    retval = fast_non_dominated_sorting(example);
    BOOST_CHECK(std::get<0>(retval) == non_dom_fronts_res);
    BOOST_CHECK(std::get<1>(retval) == dom_list_res);
    BOOST_CHECK(std::get<2>(retval) == dom_count_res);
    BOOST_CHECK(std::get<3>(retval) == non_dom_rank_res);

    // Test 3
    example = {{}, {}, {}, {}};
    non_dom_fronts_res = {{0, 1, 2, 3}};
    dom_count_res = {0, 0, 0, 0};
    dom_list_res = {{}, {}, {}, {}};
    non_dom_rank_res = {0, 0, 0, 0};

    retval = fast_non_dominated_sorting(example);
    BOOST_CHECK(std::get<0>(retval) == non_dom_fronts_res);
    BOOST_CHECK(std::get<1>(retval) == dom_list_res);
    BOOST_CHECK(std::get<2>(retval) == dom_count_res);
    BOOST_CHECK(std::get<3>(retval) == non_dom_rank_res);

    // Test 4
    example = {{0, 0, 0}};
    BOOST_CHECK_THROW(fast_non_dominated_sorting(example), std::invalid_argument);
    example = {{}};
    BOOST_CHECK_THROW(fast_non_dominated_sorting(example), std::invalid_argument);
    example = {};
    BOOST_CHECK_THROW(fast_non_dominated_sorting(example), std::invalid_argument);
    example = {{1, 3}, {3, 42, 3}, {}};
    BOOST_CHECK_THROW(fast_non_dominated_sorting(example), std::invalid_argument);
    example = {{3, 4, 5}, {}};
    BOOST_CHECK_THROW(fast_non_dominated_sorting(example), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(crowding_distance_test)
{
    std::vector<vector_double> example;
    vector_double result;
    // Test 1
    result = {2, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    example = {{0, 0}, {-1, 1}, {2, -2}};
    BOOST_CHECK(crowding_distance(example) == result);
    example = {{0.25, 0.25}, {-1, 1}, {2, -2}};
    BOOST_CHECK(crowding_distance(example) == result);
    result = {3, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    example = {{0, 0, 0}, {-1, 1, 2}, {2, -2, -2}};
    BOOST_CHECK(crowding_distance(example) == result);
    example = {{0.25, 0.25, 0.25}, {-1, 1, 2}, {2, -2, -2}};
    BOOST_CHECK(crowding_distance(example) == result);
    // Test 2
    example = {{0, 0}, {1, -1}, {2, -2}, {4, -4}};
    result = {std::numeric_limits<double>::infinity(), 1., 1.5, std::numeric_limits<double>::infinity()};
    BOOST_CHECK(crowding_distance(example) == result);
    // Test 3 - corner case
    example = {{0, 0}, {0, 0}};
    result = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    BOOST_CHECK(crowding_distance(example) == result);
    // Test 4
    example = {};
    BOOST_CHECK_THROW(crowding_distance(example), std::invalid_argument);
    example = {{}, {}};
    BOOST_CHECK_THROW(crowding_distance(example), std::invalid_argument);
    example = {{1, 2}};
    BOOST_CHECK_THROW(crowding_distance(example), std::invalid_argument);
    example = {{1}, {2}};
    BOOST_CHECK_THROW(crowding_distance(example), std::invalid_argument);
    example = {{2, 3}, {3, 4}, {2, 4, 5}};
    BOOST_CHECK_THROW(crowding_distance(example), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(sort_population_mo_test)
{
    std::vector<vector_double> example;
    std::vector<vector_double::size_type> result;
    // Test 1 - corner cases
    example = {};
    result = {};
    BOOST_CHECK(sort_population_mo(example) == result);
    example = {{1, 5, 2, 3}};
    result = {0};
    BOOST_CHECK(sort_population_mo(example) == result);
    // Test 2 - Some more complex examples
    example = {{0.25, 0.25}, {-1, 1}, {2, -2}};
    result = {1, 2, 0};
    BOOST_CHECK(sort_population_mo(example) == result);
    example = {{0, 7}, {1, 5}, {2, 3}, {4, 2}, {7, 1}, {10, 0}, {2, 6}, {4, 4}, {10, 2}, {6, 6}, {9, 5}};
    result = {0, 5, 4, 3, 1, 2, 6, 8, 7, 9, 10};
    BOOST_CHECK(sort_population_mo(example) == result);
    example = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}};
    result = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    BOOST_CHECK(sort_population_mo(example) == result);
    // Test 3 - Throws
    example = {{0}, {1, 2}};
    BOOST_CHECK_THROW(sort_population_mo(example), std::invalid_argument);
    example = {{}, {}};
    BOOST_CHECK_THROW(sort_population_mo(example), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(select_best_N_mo_test)
{
    std::vector<vector_double> example;
    std::vector<vector_double::size_type> result;
    vector_double::size_type N;

    // Test 1 - corner cases
    example = {};
    N = 10;
    result = {};
    BOOST_CHECK(select_best_N_mo(example, N) == result);
    example = {{1, 2}};
    N = 10;
    result = {0};
    BOOST_CHECK(select_best_N_mo(example, N) == result);
    example = {{1, 2}, {2, 4}, {-3, 2}, {-3, -3}};
    N = 10;
    result = {0, 1, 2, 3};
    BOOST_CHECK(select_best_N_mo(example, N) == result);
    N = 4;
    result = {0, 1, 2, 3};
    BOOST_CHECK(select_best_N_mo(example, N) == result);

    // Test 2 - The best N individuals will be a permutaion of the first N in the sorted index list.
    example = {{0, 7}, {1, 5}, {2, 3}, {4, 2}, {7, 1}, {10, 0}, {2, 6}, {4, 4}, {10, 2}, {6, 6}, {9, 5}};
    auto tmp2 = sort_population_mo(example);
    for (decltype(example.size()) i = 1; i < example.size() + 3; ++i) {
        auto tmp = select_best_N_mo(example, i);
        BOOST_CHECK(std::is_permutation(tmp.begin(), tmp.end(), tmp2.begin()));
    }
    example = {{0, 7, -2}, {1, 5, -4}, {2, 3, 1}, {4, 2, 2}, {7, 1, -10}, {10, 0, 43}};
    tmp2 = sort_population_mo(example);
    for (decltype(example.size()) i = 1; i < example.size() + 3; ++i) {
        auto tmp = select_best_N_mo(example, i);
        BOOST_CHECK(std::is_permutation(tmp.begin(), tmp.end(), tmp2.begin()));
    }
    example = {{1, 1}, {2, 2},  {-1, -1}, {1, -1},  {-1, 1}, {0, 0}, {2, 2},
               {0, 0}, {-2, 2}, {3, -2},  {-10, 2}, {-8, 4}, {4, -8}};
    tmp2 = sort_population_mo(example);
    for (decltype(example.size()) i = 1; i < example.size() + 3; ++i) {
        auto tmp = select_best_N_mo(example, i);
        BOOST_CHECK(std::is_permutation(tmp.begin(), tmp.end(), tmp2.begin()));
    }

    // Test 3 - throws
    example = {{0}, {1, 2}, {2}, {0, 0}, {6}};
    BOOST_CHECK_THROW(select_best_N_mo(example, 2u), std::invalid_argument);
    example = {{}, {}, {}, {}, {}, {}};
    BOOST_CHECK_THROW(select_best_N_mo(example, 2u), std::invalid_argument);
    example = {{1, 2}, {3, 4}, {0, 1}, {1, 0}, {2, 2}, {2, 4}};
    BOOST_CHECK_THROW(select_best_N_mo(example, 0u), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(ideal_test)
{
    std::vector<vector_double> example;
    vector_double result;
    // Test 1
    example = {};
    result = {};
    BOOST_CHECK(ideal(example) == result);
    example = {{}, {}, {}, {}, {}};
    result = {};
    BOOST_CHECK(ideal(example) == result);
    example = {{-1}, {1}, {2}, {0}, {6}};
    result = {-1};
    BOOST_CHECK(ideal(example) == result);
    example = {{1, 1}, {2, 2},  {-1, -1}, {1, -1},  {-1, 1}, {0, 0}, {2, 2},
               {0, 0}, {-2, 2}, {3, -2},  {-10, 2}, {-8, 4}, {4, -8}};
    result = {-10, -8};
    BOOST_CHECK(ideal(example) == result);
    example = {{-1, 3, 597}, {1, 2, 3645}, {2, 9, 789}, {0, 0, 231}, {6, -2, 4576}};
    result = {-1, -2, 231};
    BOOST_CHECK(ideal(example) == result);
    // Test 2 - throws
    example = {{-1}, {1, 4}, {2}, {0, 4, 2}, {6}};
    BOOST_CHECK_THROW(ideal(example), std::invalid_argument);
    example = {{}, {1}};
    BOOST_CHECK_THROW(ideal(example), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nadir_test)
{
    std::vector<vector_double> example;
    vector_double result;
    // Test 1
    example = {};
    result = {};
    BOOST_CHECK(nadir(example) == result);
    example = {{}, {}, {}, {}, {}};
    result = {};
    BOOST_CHECK(nadir(example) == result);
    example = {{-1}, {1}, {2}, {0}, {6}};
    result = {-1};
    BOOST_CHECK(nadir(example) == result);
    example = {{1, 1}, {2, 2},  {-1, -1}, {1, -1},  {-1, 1}, {0, 0}, {2, 2},
               {0, 0}, {-2, 2}, {3, -2},  {-10, 2}, {-8, 4}, {4, -8}};
    result = {4, 2};
    BOOST_CHECK(nadir(example) == result);
    example = {{0, 7}, {1, 5}, {2, 3}, {4, 2}, {7, 1}, {10, 0}, {2, 6}, {4, 4}, {12, 2}, {6, 6}, {9, 15}};
    result = {10, 7};
    BOOST_CHECK(nadir(example) == result);
    // Test 2 - throws
    example = {{-1}, {1, 4}, {2}, {0, 4, 2}, {6}};
    BOOST_CHECK_THROW(nadir(example), std::invalid_argument);
    example = {{}, {1}};
    BOOST_CHECK_THROW(nadir(example), std::invalid_argument);
}

void check_weights(const std::vector<std::vector<double>> &win, vector_double::size_type n_f)
{
    for (auto lambda : win) {
        BOOST_CHECK_EQUAL(lambda.size(), n_f);
        auto sum = std::accumulate(lambda.begin(), lambda.end(), 0.);
        BOOST_CHECK_CLOSE(sum, 1., 1e-08);
    }
}

BOOST_AUTO_TEST_CASE(decomposition_weights_test)
{
    detail::random_engine_type r_engine(23u);
    // We test some throws
    // At least 2 objectives are needed
    BOOST_CHECK_THROW(decomposition_weights(1u, 5u, "grid", r_engine), std::invalid_argument);
    // The weight generation method must be one of 'grid', 'random', 'low discrepancy'
    BOOST_CHECK_THROW(decomposition_weights(2u, 5u, "grod", r_engine), std::invalid_argument);
    // The number of weights are smaller than the number of objectives
    BOOST_CHECK_THROW(decomposition_weights(10u, 5u, "grid", r_engine), std::invalid_argument);
    // The number of weights is not compatible with 'grid'
    BOOST_CHECK_THROW(decomposition_weights(4u, 31u, "grid", r_engine), std::invalid_argument);

    // We test some known cases
    {
        auto ws = decomposition_weights(3u, 3u, "grid", r_engine);
        check_weights(ws, 3u);
    }
    {
        auto ws = decomposition_weights(3u, 3u, "random", r_engine);
        check_weights(ws, 3u);
    }
    {
        auto ws = decomposition_weights(3u, 3u, "low discrepancy", r_engine);
        check_weights(ws, 3u);
    }
    {
        auto ws = decomposition_weights(3u, 6u, "grid", r_engine);
        check_weights(ws, 3u);
    }
    {
        auto ws = decomposition_weights(3u, 4u, "random", r_engine);
        check_weights(ws, 3u);
    }
    {
        auto ws = decomposition_weights(3u, 4u, "low discrepancy", r_engine);
        check_weights(ws, 3u);
    }
    {
        auto ws = decomposition_weights(2u, 4u, "grid", r_engine);
        check_weights(ws, 2u);
    }
    {
        auto ws = decomposition_weights(2u, 4u, "random", r_engine);
        check_weights(ws, 2u);
    }
    {
        auto ws = decomposition_weights(2u, 4u, "low discrepancy", r_engine);
        check_weights(ws, 2u);
    }
    {
        auto ws = decomposition_weights(5u, 25u, "random", r_engine);
        check_weights(ws, 5u);
    }
}

BOOST_AUTO_TEST_CASE(decompose_objectives_test)
{
    vector_double weight{0.5, 0.5};
    vector_double ref_point{0., 0.};
    vector_double f{1.234, -1.345};
    auto fw = decompose_objectives(f, weight, ref_point, "weighted")[0];
    auto ft = decompose_objectives(f, weight, ref_point, "tchebycheff")[0];
    auto fb = decompose_objectives(f, weight, ref_point, "bi")[0];

    BOOST_CHECK_CLOSE(f[0] * weight[0] + f[1] * weight[1], fw, 1e-8);
    BOOST_CHECK_CLOSE(std::max(weight[0] * std::abs(f[0] - ref_point[0]), weight[1] * std::abs(f[1] - ref_point[1])),
                      ft, 1e-8);
    double lnorm = std::sqrt(weight[0] * weight[0] + weight[1] * weight[1]);
    vector_double ilambda{weight[0] / lnorm, weight[1] / lnorm};
    double d1 = (f[0] - ref_point[0]) * ilambda[0] + (f[1] - ref_point[1]) * ilambda[1];
    double d20 = f[0] - (ref_point[0] + d1 * ilambda[0]);
    double d21 = f[1] - (ref_point[1] + d1 * ilambda[1]);
    d20 *= d20;
    d21 *= d21;
    double d2 = std::sqrt(d20 + d21);
    BOOST_CHECK_CLOSE(d1 + 5.0 * d2, fb, 1e-8);

    // We check the throws
    BOOST_CHECK_THROW(decompose_objectives(f, {1., 2., 3., 4.}, ref_point, "weighted"), std::invalid_argument);
    BOOST_CHECK_THROW(decompose_objectives(f, weight, {1.}, "weighted"), std::invalid_argument);
    BOOST_CHECK_THROW(decompose_objectives(f, weight, ref_point, "pippo"), std::invalid_argument);
    BOOST_CHECK_THROW(decompose_objectives({}, {}, {}, "weighted"), std::invalid_argument);
}
