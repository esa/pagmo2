#define BOOST_TEST_MODULE pagmo_null_problem_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>


#include "../include/algorithms/sea.hpp"
#include "../include/io.hpp"
#include "../include/problems/inventory.hpp"
#include "../include/population.hpp"
#include "../include/types.hpp"

using namespace pagmo;
using namespace std;

BOOST_AUTO_TEST_CASE(sea_algorithm_test)
{
    problem prob{inventory{25u, 5u, 32u}};
    population pop{prob, 5u, 23u};
    algorithm algo(sea{1000u, 23u});
    algo.set_verbosity(1u);
    print(prob, '\n', algo, '\n');
    pop = algo.evolve(pop);
}
