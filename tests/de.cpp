#define BOOST_TEST_MODULE pagmo_null_problem_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include "../include/algorithm.hpp"
#include "../include/algorithms/de.hpp"
#include "../include/algorithms/null_algorithm.hpp"
#include "../include/io.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/population.hpp"
#include "../include/serialization.hpp"
#include "../include/types.hpp"

using namespace pagmo;
using namespace std;

BOOST_AUTO_TEST_CASE(de_algorithm_construction)
{
    de user_algo{10000u, 0.7, 0.5, 2u, 1e-6, 1e-6, 23u};
    problem prob{rosenbrock{10u}};
    population pop{prob, 20u, 23u};
    user_algo.set_verbosity(100u);
    pop = user_algo.evolve(pop);
}
