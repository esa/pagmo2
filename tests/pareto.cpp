#define BOOST_TEST_MODULE pagmo_pareto_utilities_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>
#include <tuple>

#include "../include/utils/pareto.hpp"
#include "../include/types.hpp"
#include "../include/io.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(pareto_dominance_test)
{
    BOOST_CHECK(pareto_dominance({1,2,3},{4,5,6}));
    BOOST_CHECK(!pareto_dominance({4,5,6},{4,5,6}));
    BOOST_CHECK(pareto_dominance({4,5,5},{4,5,6}));
    BOOST_CHECK(!pareto_dominance({1,2,3},{2,1,5}));
    BOOST_CHECK(pareto_dominance({-3.4,1.5,2.9,-2.3,4.99,3.2,6.6},{1,2,3,4,5,6,7}));
}

BOOST_AUTO_TEST_CASE(fast_non_dominated_sorting_test)
{
    // We create various values to compute
    std::vector<vector_double::size_type>               dom_count;
    std::vector<std::vector<vector_double::size_type>>  dom_list;
    std::vector<vector_double::size_type>               non_dom_rank;

    // And i/o values to check
    std::vector<vector_double> example;
    std::vector<std::vector<vector_double::size_type>>  non_dom_fronts_res;
    std::vector<vector_double::size_type>               dom_count_res;
    std::vector<std::vector<vector_double::size_type>>  dom_list_res;
    std::vector<vector_double::size_type>               non_dom_rank_res;

    // Test 1
    example = {{0,7},{1,5},{2,3},{4,2},{7,1},{10,0},{2,6},{4,4},{10,2},{6,6},{9,5}};
    non_dom_fronts_res = {{0,1,2,3,4,5},{6,7,8},{9,10}};
    dom_count_res = {0,0,0,0,0,0,2,2,3,5,5};
    dom_list_res = {{},{6,9,10},{6,7,9,10}, {7,8,9,10}, {8,10}, {8}, {9}, {9,10},{},{},{}};
    non_dom_rank_res = {0,0,0,0,0,0,1,1,1,2,2};

    auto retval = fast_non_dominated_sorting(example);
    BOOST_CHECK(std::get<0>(retval) == non_dom_fronts_res);
    BOOST_CHECK(std::get<1>(retval) == dom_list_res);
    BOOST_CHECK(std::get<2>(retval) == dom_count_res);
    BOOST_CHECK(std::get<3>(retval) == non_dom_rank_res);

    // Test 2
    example = {{1,2,3},{-2,3,7},{-1,-2,-3},{0,0,0}};
    non_dom_fronts_res = {{1,2},{3},{0}};
    dom_count_res = {2, 0, 0, 1};
    dom_list_res = {{},{},{0,3},{0}};
    non_dom_rank_res = {2,0,0,1};

    retval = fast_non_dominated_sorting(example);
    BOOST_CHECK(std::get<0>(retval) == non_dom_fronts_res);
    BOOST_CHECK(std::get<1>(retval) == dom_list_res);
    BOOST_CHECK(std::get<2>(retval) == dom_count_res);
    BOOST_CHECK(std::get<3>(retval) == non_dom_rank_res);

}