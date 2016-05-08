#define BOOST_TEST_MODULE discrepancy_test
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <exception>
#include <tuple>

#include "../include/utils/discrepancy.hpp"
#include "../include/types.hpp"
#include "../include/io.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(van_der_corput_test)
{   
    // We test explicitly the first ten elements of the Van der Corput
    // sequences corresponding to base 2 and base 10.
    std::vector<double> computed2;
    std::vector<double> computed10;
    for (auto i = 0u; i<10;++i) {
        computed2.push_back(van_der_corput(i, 2u));
        computed10.push_back(van_der_corput(i, 10u));
    }
    std::vector<double> real2{0.,0.5,0.25,0.75,0.125,0.625,0.375,0.875,0.0625,0.5625};
    std::vector<double> real10{0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
    BOOST_CHECK(real2 == computed2); // in base 2 no need for approximate comparison as all members are represented correctly
    for (auto i = 0u; i<10;++i) {     // in base 10 we need to check with a tolerance as per floating point representation problems
        BOOST_CHECK_CLOSE(real10[i], computed10[i], 1e-13);
    }
}