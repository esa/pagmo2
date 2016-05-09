#define BOOST_TEST_MODULE discrepancy_test
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <exception>
#include <tuple>

#include "../include/utils/discrepancy.hpp"
#include "../include/types.hpp"
#include "../include/io.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(sample_from_simplex_test)
{
    auto case1 = sample_from_simplex({0.3,0.1,0.6,0.9,1.});
    auto result1 = std::vector<double>{0.1,0.2,0.3,0.3,0.1,0.};
    for (auto i = 0u; i < case1.size(); ++i) { 
        BOOST_CHECK_CLOSE(case1[i], result1[i], 1e-13);
    }

    auto case2 = sample_from_simplex({0.,0.9,0.3,1.,1.,0.2,0.});
    auto result2 = std::vector<double>{0.,0.,0.2,0.1,0.6,0.1,0.,0.};
    for (auto i = 0u; i < case2.size(); ++i) { 
        BOOST_CHECK_CLOSE(case2[i], result2[i], 1e-13);
    }

    auto case3 = sample_from_simplex({0.2});
    auto result3 = std::vector<double>{0.2,0.8};
    for (auto i = 0u; i < case3.size(); ++i) { 
        BOOST_CHECK_CLOSE(case3[i], result3[i], 1e-13);
    }

    // Check that throws if point is not in [0,1]
    BOOST_CHECK_THROW(sample_from_simplex({0.2, 2.3}), std::invalid_argument);
    BOOST_CHECK_THROW(sample_from_simplex({0.3,0.1,0.6,0.9,-0.1,1.}), std::invalid_argument);
    // Checks that input cannot be empty
    BOOST_CHECK_THROW(sample_from_simplex({}), std::invalid_argument);
}

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

BOOST_AUTO_TEST_CASE(halton_test)
{
    std::vector<std::vector<double>> computed2dim;
    for (auto i = 0u; i<6u;++i) {
        computed2dim.push_back(halton(i, 2u));
    }
    std::vector<std::vector<double>> real2dim{{0.,0.}, {1./2.,1./3.},{1./4., 2./3.},{3./4.,1./9.},{1./8.,4./9.}, {5./8.,7./9.}};
    for (auto i = 0u; i<6u;++i) {     // in base 10 we need to check with a tolerance as per floating point representation problems
        for (auto j = 0u; j<2u;++j) { 
            BOOST_CHECK_CLOSE(real2dim[i][j], computed2dim[i][j], 1e-13);
        }
    }
}