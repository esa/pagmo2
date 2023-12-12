#define BOOST_TEST_MODULE nsga3_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga3.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(nsga3_instance){
    BOOST_CHECK_NO_THROW(nsga3{});
};
