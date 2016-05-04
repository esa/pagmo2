#define BOOST_TEST_MODULE generic_utilities_test
#include <boost/test/unit_test.hpp>
#include <exception>
#include <tuple>

#include "../include/utils/generic.hpp"
#include "../include/types.hpp"
#include "../include/io.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(decision_vector_test)
{   
    auto inf = std::numeric_limits<double>::infinity();
    auto big = std::numeric_limits<double>::max();

    // Test the throws
    BOOST_CHECK_THROW(decision_vector({{1,2},{0,3}}), std::invalid_argument);
    BOOST_CHECK_THROW(decision_vector({{1,-big},{0,big}}), std::invalid_argument);
    BOOST_CHECK_THROW(decision_vector({{1,-inf},{0,32}}), std::invalid_argument);
    BOOST_CHECK_THROW(decision_vector({{1,2,3},{0,3}}), std::invalid_argument);

    // Test the results
    BOOST_CHECK((decision_vector({{3,4},{3,4}}) == vector_double{3,4}));
    BOOST_CHECK(decision_vector({{0,0},{1,1}})[0] >= 0);
    BOOST_CHECK(decision_vector({{0,0},{1,1}})[1] < 1);

    // Test the overload
    BOOST_CHECK(decision_vector({0,0},{1,1})[0] >= 0);
    BOOST_CHECK(decision_vector({0,0},{1,1})[1] < 1);
}

