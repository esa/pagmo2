#define BOOST_TEST_MODULE moead_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include "../include/algorithm.hpp"
#include "../include/algorithms/moead.hpp"
#include "../include/io.hpp"
#include "../include/serialization.hpp"
#include "../include/types.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(moead_construction)
{
    moead user_algo{};
    algorithm algo{user_algo};
    print(algo,'\n');
}
