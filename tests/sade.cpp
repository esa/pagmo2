#define BOOST_TEST_MODULE sade_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include "../include/algorithm.hpp"
#include "../include/algorithms/sade.hpp"
#include "../include/algorithms/de.hpp"
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

    vector_double res, res1;
    sade user_algo{500u, 2u, 1u, 1e-30, 1e-30};
    for (auto i = 0u; i< 1000u; ++i){
        population pop{prob, 20u};
        pop = user_algo.evolve(pop);
        res.push_back(pop.get_f()[pop.best_idx()][0]);
    }
    print("sade: ", std::accumulate(res.begin(), res.end(), 0.) / static_cast<double>(res.size()),'\n');

    de user_algo2{500u, 0.8, 0.9, 2u, 1e-30, 1e-30};
    for (auto i = 0u; i< 1000u; ++i){
        population pop{prob, 20u};
        pop = user_algo2.evolve(pop);
        res1.push_back(pop.get_f()[pop.best_idx()][0]);
    }
    print("de: ", std::accumulate(res1.begin(), res1.end(), 0.) / static_cast<double>(res1.size()),'\n');

}
