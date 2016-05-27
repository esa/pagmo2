#define BOOST_TEST_MODULE sade_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include "../include/algorithm.hpp"
#include "../include/algorithms/sade.hpp"
#include "../include/algorithms/null_algorithm.hpp"
#include "../include/io.hpp"
#include "../include/population.hpp"
#include "../include/problems/hock_schittkowsky_71.hpp"
#include "../include/problems/inventory.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/problems/zdt.hpp"
#include "../include/serialization.hpp"
#include "../include/types.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(sade_evolve_test)
{
    problem prob{rosenbrock{10u}};
    population pop{prob, 20u};

    sade user_algo{500u};
    user_algo.set_verbosity(1u);
    pop = user_algo.evolve(pop);
}
