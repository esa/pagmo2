#define BOOST_TEST_MODULE pagmo_problem_test
#include <boost/test/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <sstream>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "../include/algorithms/null_algorithm.hpp"
#include "../include/serialization.hpp"
#include "../include/types.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(algorithm_stochastic_test)
{
    algorithm algo{null_algorithm{}};
    print(algo);
}
