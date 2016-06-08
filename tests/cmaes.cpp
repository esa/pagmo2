#define BOOST_TEST_MODULE sade_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include "../include/algorithms/cmaes.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/problems/rastrigin.hpp"
#include "../include/population.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(convergence_test)
{
    {
    // We test that CMA-ES is able to converge in rosenbrock{10u} within 11000 fevals
    cmaes user_algo{500u};
    rosenbrock prob{10u};
    population pop(prob, 20u);
    pop = user_algo.evolve(pop);
    BOOST_CHECK(pop.get_problem().get_fevals() < 11000u);
    }
    {
    // We test that CMA-ES is able to converge in rastrigin{10u} within 11000 fevals
    cmaes user_algo{500u};
    user_algo.set_verbosity(1u);
    rastrigin prob{10u};
    population pop(prob, 20u);
    pop = user_algo.evolve(pop);
    BOOST_CHECK(pop.get_problem().get_fevals() < 4000u);
    }
}
