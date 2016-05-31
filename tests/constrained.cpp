#define BOOST_TEST_MODULE constrained_utilities_test
#include <boost/test/unit_test.hpp>
#include <exception>

#include "../include/utils/constrained.hpp"
#include "../include/types.hpp"
#include "../include/io.hpp"

using namespace pagmo;


BOOST_AUTO_TEST_CASE(sort_population_con_test)
{
    std::vector<vector_double> example;
    vector_double::size_type neq;
    vector_double tol;
    std::vector<vector_double::size_type> result;
    // Test 1
    example = {{0,0,0},{1,1,0},{2,0,0}};
    neq = 1;
    result = {0, 2, 1};
    tol = {0.,0.};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) == result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{0,0,0},{1,0,0},{2,0,0}};
    neq = 1;
    result = {0, 1, 2};
    tol = {0.,0.};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) == result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{-1,0,-20},{0,0,-1},{1,0,-2}};
    neq = 1;
    result = {0, 1, 2};
    tol = {0.,0.};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) == result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{-1,0,-20},{0,0,-1},{1,0,-2}};
    neq = 2;
    result = {1, 2, 0};
    tol = {0.,0.};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) == result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{-1,0,0},{0,0,-1},{1,0,0}};
    neq = 2;
    result = {0, 1, 2};
    tol = {0.,1.};
    BOOST_CHECK(sort_population_con(example, neq) != result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) != result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{-1,0,-20},{0,0,-1},{1,0,-2}};
    neq = 0;
    result = {0, 1, 2};
    tol = {0.,0.};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) == result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{1},{0},{2},{3}};
    neq = 0;
    result = {1, 0, 2, 3};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    // Test throw
    example = {{1,2,3},{1,2}};
    BOOST_CHECK_THROW(sort_population_con(example, neq), std::invalid_argument);
    example = {{-1,0,0},{0,0,-1},{1,0,0}};
    BOOST_CHECK_THROW(sort_population_con(example, 3), std::invalid_argument);
    BOOST_CHECK_THROW(sort_population_con(example, 4), std::invalid_argument);
    tol = {2,3,4};
    BOOST_CHECK_THROW(sort_population_con(example, 0, tol), std::invalid_argument);
    tol = {2};
    BOOST_CHECK_THROW(sort_population_con(example, 0, tol), std::invalid_argument);
}
