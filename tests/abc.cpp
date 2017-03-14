#define BOOST_TEST_MODULE abc_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
//#include <pagmo/algorithms/abc.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/algorithms/null_algorithm.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/ackley.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(abc_algorithm_construction)
{
    algorithm algo{de{100u, 0.8, 0.8, 2u, 1E-6, 1E-6, 32u}};
    algo.set_verbosity(10u);
    population pop{ackley{10u}, 20u, 32u};
    pop = algo.evolve(pop);

    population pop2{ackley{10u}, 20u, 32u};
    algo.set_seed(32u);
    pop2 = algo.evolve(pop2);
}

BOOST_AUTO_TEST_CASE(abc_evolve_test)
{
}

BOOST_AUTO_TEST_CASE(abc_setters_getters_test)
{
}

BOOST_AUTO_TEST_CASE(abc_serialization_test)
{
}
