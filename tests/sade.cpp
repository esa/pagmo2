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

BOOST_AUTO_TEST_CASE(sade_algorithm_construction)
{
    sade user_algo{53u, 2u, 1u, 1e-6, 1e-6, false, 23u};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK((user_algo.get_log() == sade::log_type{}));

    BOOST_CHECK_THROW((sade{53u, 0u,  1u, 1e-6, 1e-6, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((sade{53u, 23u, 1u, 1e-6, 1e-6, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((sade{53u, 2u,  0u, 1e-6, 1e-6, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((sade{53u, 2u,  3u, 1e-6, 1e-6, false, 23u}), std::invalid_argument);
}
