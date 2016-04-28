#define BOOST_TEST_MODULE pagmo_constrained_utilities_test
#include <boost/test/unit_test.hpp>

#include "../include/utils/constrained.hpp"
#include "../include/types.hpp"
#include "../include/io.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(test_eq_constraints_test)
{
    // Test 1
    vector_double example;
    std::pair<vector_double::size_type, double> result;
    example = {};
    result = {0u,0.};
    BOOST_CHECK(test_eq_constraints(example) == result);
    example = {0., 0., 0.};
    result = {3u,0.};
    BOOST_CHECK(test_eq_constraints(example) == result);
    example = {1e-4, 0., 0.};
    result = {2u,1e-4};
    BOOST_CHECK(test_eq_constraints(example) == result);
    example = {.5, .5, .5, .5};
    result = {0u,1.};
    BOOST_CHECK(test_eq_constraints(example) == result);
    example = {0., 0., 0.};
    result = {3u,0.};
    BOOST_CHECK(test_eq_constraints(example,{1e-1,1e-3,1e-5}) == result);
    example = {1e-4, 0., 0.};
    result = {3u,0.};
    BOOST_CHECK(test_eq_constraints(example, {1e-2,0.,0.}) == result);
    example = {.5, .5, .5, .5};
    result = {3u,.5};
    BOOST_CHECK(test_eq_constraints(example, {0,.5,3.,4.}) == result);
    example = {0., 0., 0.};
    result = {3u,0.};
    BOOST_CHECK(test_eq_constraints(example,1e-2) == result);
    example = {1e-4, 0., 0.};
    result = {3u,0.};
    BOOST_CHECK(test_eq_constraints(example, 1e-2) == result);
    example = {.5, .5, .5, .5};
    result = {4u,0.};
    BOOST_CHECK(test_eq_constraints(example, 0.5) == result);
    example = {1., 2., 3., 4.};
    result = {2u,std::sqrt(5.)};
    BOOST_CHECK(test_eq_constraints(example, 2.) == result);
    // Test 2 - throws
    example = {1e-4, 0., 0.};
    BOOST_CHECK_THROW(test_eq_constraints(example, {2,3}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_ineq_constraints_test)
{
    // Test 1
    vector_double example;
    std::pair<vector_double::size_type, double> result;
    example = {};
    result = {0u,0.};
    BOOST_CHECK(test_ineq_constraints(example) == result);
    example = {-1, 2., -3};
    result = {2u,2.};
    BOOST_CHECK(test_ineq_constraints(example) == result);
    example = {-1, 2., -3};
    result = {2u,1.};
    BOOST_CHECK(test_ineq_constraints(example, {1.,1.,1.}) == result);
    example = {-1, 2., -3};
    result = {3u,0.};
    BOOST_CHECK(test_ineq_constraints(example, {2.,2.,2.}) == result);
    example = {-1, 2., -3, 3., -1e-4, 1e-4};
    result = {4u, std::sqrt(5)};
    BOOST_CHECK(test_ineq_constraints(example, {1.,1.,1.,1.,1.,1.}) == result);
    example = {-1, 2., -3, 3., -1e-4, 1e-4};
    result = {4u, std::sqrt(2)};
    BOOST_CHECK(test_ineq_constraints(example, {1.,1.,1.,2.,1.,1.}) == result);
    example = {-1, 2., -3};
    result = {3u,0.};
    BOOST_CHECK(test_ineq_constraints(example, 2.) == result);
    example = {-1, 2., -3, 3., -1e-4, 1e-4};
    result = {4u, std::sqrt(5)};
    BOOST_CHECK(test_ineq_constraints(example, 1.) == result);
    // Test 2 - throws
    example = {1e-4, 0., 0.};
    BOOST_CHECK_THROW(test_ineq_constraints(example, {2,3}), std::invalid_argument);
}
