#define BOOST_TEST_MODULE pagmo_null_problem_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>


#include "../include/algorithms/sea.hpp"
#include "../include/io.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/population.hpp"
#include "../include/types.hpp"

using namespace pagmo;
using namespace std;

BOOST_AUTO_TEST_CASE(sea_algorithm_test)
{
    problem prob{rosenbrock{2u}};
    population pop{prob, 5u, 23u};
    algorithm algo(sea{1000u, 23u});
    algo.set_verbosity(1u);
    print(algo, '\n');
    pop = algo.evolve(pop);
    for ( const auto& i : algo.extract<sea>()->get_log()) {
      cout << get<0>(i) << ", " << get<1>(i) << ", " << get<2>(i) << ", " << get<3>(i) << endl;
    }
}
