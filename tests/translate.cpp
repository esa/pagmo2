#define BOOST_TEST_MODULE pagmo_translate_test
#include <boost/test/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <stdexcept>

#include "../include/types.hpp"
#include "../include/problems/translate.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(translate_construction)
{
    translate p0{};
    translate p1{null{},{1}};
    auto p0_string = boost::lexical_cast<std::string>(p0);
    auto p1_string = boost::lexical_cast<std::string>(p1);

    // We check that the default constructor constructs a problem
    // which has an identical representation to the problem
    // built by the explicit constructor.
    BOOST_CHECK(p0_string==p1_string);

    // We check that wrong size for translation results in a problem
    BOOST_CHECK_THROW((translate{null{},{1,2}}), std::invalid_argument);

}
