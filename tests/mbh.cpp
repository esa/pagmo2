#define BOOST_TEST_MODULE mbh_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/mbh.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(mbh_algorithm_construction)
{
    compass_search{10u}.evolve(population{problem{hock_schittkowsky_71{}}, 15u});
    mbh user_algo{compass_search{}, 50u, 0.1};
    algorithm algo{user_algo};
    std::cout << algo << "\n";
    problem prob{hock_schittkowsky_71{}};
    population pop{prob, 1u};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);
}
