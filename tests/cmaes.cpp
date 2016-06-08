#define BOOST_TEST_MODULE cmaes_test
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

BOOST_AUTO_TEST_CASE(cmaes_algorithm_construction)
{
    de user_algo{1234u, 0.7, 0.5, 2u, 1e-6, 1e-6, 23u};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK((user_algo.get_log() == de::log_type{}));

    BOOST_CHECK_THROW((de{1234u, 1.2}), std::invalid_argument);
    BOOST_CHECK_THROW((de{1234u,-0.4}), std::invalid_argument);
    BOOST_CHECK_THROW((de{1234u, 0.7, 1.2}), std::invalid_argument);
    BOOST_CHECK_THROW((de{1234u, 0.7,-1.2}), std::invalid_argument);
    BOOST_CHECK_THROW((de{1234u, 0.7, 0.5, 12u}), std::invalid_argument);
}
