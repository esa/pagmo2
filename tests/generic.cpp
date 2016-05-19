#define BOOST_TEST_MODULE generic_utilities_test
#include <boost/test/unit_test.hpp>
#include <exception>
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
    detail::random_engine_type r_engine(pagmo::random_device::next());

    // Test the throws
    BOOST_CHECK_THROW(uniform_real_from_range(1,0, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-big, big, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-3, inf, r_engine), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(decision_vector_test)
{
    auto inf = std::numeric_limits<double>::infinity();
    auto big = std::numeric_limits<double>::max();
    detail::random_engine_type r_engine(pagmo::random_device::next());

    // Test the throws
    BOOST_CHECK_THROW(decision_vector({{1,2},{0,3}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(decision_vector({{1,-big},{0,big}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(decision_vector({{1,-inf},{0,32}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(decision_vector({{1,2,3},{0,3}}, r_engine), std::invalid_argument);

    // Test the results
    BOOST_CHECK((decision_vector({{3,4},{3,4}}, r_engine) == vector_double{3,4}));
    BOOST_CHECK(decision_vector({{0,0},{1,1}}, r_engine)[0] >= 0);
    BOOST_CHECK(decision_vector({{0,0},{1,1}}, r_engine)[1] < 1);

    // Test the overload
    BOOST_CHECK((decision_vector({3,4},{3,4}, r_engine) == vector_double{3,4}));
    BOOST_CHECK(decision_vector({0,0},{1,1}, r_engine)[0] >= 0);
    BOOST_CHECK(decision_vector({0,0},{1,1}, r_engine)[1] < 1);
}
