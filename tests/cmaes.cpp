#define BOOST_TEST_MODULE sade_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include "../include/algorithms/cmaes.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(construction_test)
{
    cmaes user_algo{};
}
