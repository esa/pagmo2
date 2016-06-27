#define BOOST_TEST_MODULE generic_utilities_test
#include <boost/test/included/unit_test.hpp>

#include <exception>
#include <limits>
#include <tuple>

#include "../include/utils/generic.hpp"
#include "../include/rng.hpp"
#include "../include/types.hpp"
#include "../include/io.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(uniform_real_from_range_test)
{
    auto inf = std::numeric_limits<double>::infinity();
    auto big = std::numeric_limits<double>::max();
    auto nan = std::numeric_limits<double>::quiet_NaN();
    detail::random_engine_type r_engine(pagmo::random_device::next());

    // Test the throws
    BOOST_CHECK_THROW(uniform_real_from_range(1,0, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-big, big, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-3, inf, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-nan, nan, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(nan, nan, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-nan, 3, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-3, nan, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(inf, inf, r_engine), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(decision_vector_test)
{
    auto inf = std::numeric_limits<double>::infinity();
    auto big = std::numeric_limits<double>::max();
    auto nan = std::numeric_limits<double>::quiet_NaN();
    detail::random_engine_type r_engine(pagmo::random_device::next());

    // Test the throws
    BOOST_CHECK_THROW(decision_vector({{1,2},{0,3}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(decision_vector({{1,-big},{0,big}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(decision_vector({{1,-inf},{0,32}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(decision_vector({{1,2,3},{0,3}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(decision_vector({{0,2,3},{1,4,nan}}, r_engine), std::invalid_argument);

    // Test the results
    BOOST_CHECK((decision_vector({{3,4},{3,4}}, r_engine) == vector_double{3,4}));
    BOOST_CHECK(decision_vector({{0,0},{1,1}}, r_engine)[0] >= 0);
    BOOST_CHECK(decision_vector({{0,0},{1,1}}, r_engine)[1] < 1);

    // Test the overload
    BOOST_CHECK((decision_vector({3,4},{3,4}, r_engine) == vector_double{3,4}));
    BOOST_CHECK(decision_vector({0,0},{1,1}, r_engine)[0] >= 0);
    BOOST_CHECK(decision_vector({0,0},{1,1}, r_engine)[1] < 1);
}

BOOST_AUTO_TEST_CASE(force_bounds_test)
{
    detail::random_engine_type r_engine(32u);
    // force_bounds_random
    {
    vector_double x{1.,2.,3.};
    vector_double x_fix = x;
    detail::force_bounds_random(x_fix, {0., 0., 0.}, {3.,3.,3.}, r_engine);
    BOOST_CHECK(x == x_fix);
    detail::force_bounds_random(x_fix, {0., 0., 0.}, {1.,1.,1.}, r_engine);
    BOOST_CHECK(x != x_fix);
    BOOST_CHECK_EQUAL(x_fix[0], 1.);
    BOOST_CHECK(x_fix[1] <= 1. && x_fix[1] >= 0.);
    BOOST_CHECK(x_fix[2] <= 1. && x_fix[2] >= 0.);
    }
    // force_bounds_reflection
    {
    vector_double x{1.,2.,5.};
    vector_double x_fix = x;
    detail::force_bounds_reflection(x_fix, {0., 0., 0.}, {3.,3.,5.});
    BOOST_CHECK(x == x_fix);
    detail::force_bounds_reflection(x_fix, {0., 0., 0.}, {1., 1.9, 2.1});
    BOOST_CHECK(x != x_fix);
    BOOST_CHECK_EQUAL(x_fix[0], 1.);
    BOOST_CHECK_CLOSE(x_fix[1], 1.8, 1e-8);
    BOOST_CHECK_CLOSE(x_fix[2], 0.8, 1e-8);
    }
    // force_bounds_stick
    {
    vector_double x{1.,2.,5.};
    vector_double x_fix = x;
    detail::force_bounds_stick(x_fix, {0., 0., 0.}, {3.,3.,5.});
    BOOST_CHECK(x == x_fix);
    // ub
    detail::force_bounds_stick(x_fix, {0., 0., 0.}, {1., 1.9, 2.1});
    BOOST_CHECK(x != x_fix);
    BOOST_CHECK_EQUAL(x_fix[0], 1.);
    BOOST_CHECK_EQUAL(x_fix[1], 1.9);
    BOOST_CHECK_EQUAL(x_fix[2], 2.1);
    // lb
    detail::force_bounds_stick(x_fix, {2., 2., 2.}, {3., 3., 3.});
    BOOST_CHECK_EQUAL(x_fix[0], 2.);
    BOOST_CHECK_EQUAL(x_fix[1], 2.);
    BOOST_CHECK_EQUAL(x_fix[2], 2.1);

    }
}

BOOST_AUTO_TEST_CASE(safe_cast_test)
{
    unsigned short s = std::numeric_limits<unsigned short>::max();
    unsigned long l = std::numeric_limits<unsigned long>::max();
    BOOST_CHECK_NO_THROW(safe_cast<unsigned long>(s));
    if (l > s) {
        BOOST_CHECK_THROW(safe_cast<unsigned short>(l), std::overflow_error);
    }
}

BOOST_AUTO_TEST_CASE(binomial_coefficient_test)
{
    BOOST_CHECK_EQUAL(binomial_coefficient(0u,0u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(1u,0u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(1u,1u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(2u,0u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(2u,1u), 2u);
    BOOST_CHECK_EQUAL(binomial_coefficient(2u,2u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(13u,5u), 1287u);
    BOOST_CHECK_EQUAL(binomial_coefficient(21u,10u), 352716u);
    BOOST_CHECK_THROW(binomial_coefficient(10u,21u), std::invalid_argument);
    BOOST_CHECK_THROW(binomial_coefficient(0u,1u), std::invalid_argument);
    BOOST_CHECK_THROW(binomial_coefficient(4u,7u), std::invalid_argument);
}
