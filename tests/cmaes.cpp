#define BOOST_TEST_MODULE sade_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include "../include/algorithms/cmaes.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/population.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(construction_test)
{
    cmaes user_algo{500u};
    rosenbrock prob{10u};
    population pop(prob, 20u);
    print(pop.get_f()[pop.best_idx()], '\n');
    pop = user_algo.evolve(pop);
    print(pop.get_f()[pop.best_idx()], '\n');
    print(pop.get_problem().get_fevals(), '\n');
}
