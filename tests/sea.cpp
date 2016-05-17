#define BOOST_TEST_MODULE pagmo_null_problem_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>


#include "../include/algorithms/sea.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/population.hpp"
#include "../include/types.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(sea_algorithm_test)
{
    problem prob{rosenbrock{5u}};
    population pop{prob, 5u};
    algorithm algo(sea{1000u});
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);
}
