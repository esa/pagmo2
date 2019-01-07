/* Copyright 2017 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#define BOOST_TEST_MODULE gi_aco_mo_test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/giACOmo.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(gi_aco_mo_algorithm_construction)
{
    gi_aco_mo user_algo{1u, 0.95, 10, 20, 30, 0.95, 40, 50., 60, 0.95, 32u};
    BOOST_CHECK_NO_THROW(gi_aco_mo{});
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 32u);
    // BOOST_CHECK((user_algo.get_log() == moead::log_type{}));

    // Check the throws
    // Wrong acc
    BOOST_CHECK_THROW((gi_aco_mo{1u, 1.1, 10, 20, 30, 0.95, 40, 50., 60, 0.95, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((gi_aco_mo{1u, -1.1, 10, 20, 30, 0.95, 40, 50., 60, 0.95, 32u}), std::invalid_argument);
    // Wrong FSTOP
    BOOST_CHECK_THROW((gi_aco_mo{1u, 0.95, -1, 20, 30, 0.95, 40, 50., 60, 0.95, 32u}), std::invalid_argument);
    // Wrong IMPSTOP
    BOOST_CHECK_THROW((gi_aco_mo{1u, 0.95, 10, -1, 30, 0.95, 40, 50., 60, 0.95, 32u}), std::invalid_argument);
    // Wrong EVALSTOP
    BOOST_CHECK_THROW((gi_aco_mo{1u, 0.95, 10, 20, -1, 0.95, 40, 50., 60, 0.95, 32u}), std::invalid_argument);
    // Wrong FOCUS
    BOOST_CHECK_THROW((gi_aco_mo{1u, 0.95, 10, 20, 30, 1.1, 40, 50., 60, 0.95, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((gi_aco_mo{1u, 0.95, 10, 20, 30, -0.1, 40, 50., 60, 0.95, 32u}), std::invalid_argument);
    // Wrong KER
    BOOST_CHECK_THROW((gi_aco_mo{1u, 0.95, 10, 20, 30, 0.95, -1, 50., 60, 0.95, 32u}), std::invalid_argument);
    // Wrong ORACLE
    BOOST_CHECK_THROW((gi_aco_mo{1u, 0.95, 10, 20, 30, 0.95, 40, -0.1, 60, 0.95, 32u}), std::invalid_argument);
    // Wrong PARETOMAX
    BOOST_CHECK_THROW((gi_aco_mo{1u, 0.95, 10, 20, 30, 0.95, 40, 50., -1, 0.95, 32u}), std::invalid_argument);
    // Wrong EPSILON
    BOOST_CHECK_THROW((gi_aco_mo{1u, 0.95, 10, 20, 30, 0.95, 40, 50., 60, 1.1, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((gi_aco_mo{1u, 0.95, 10, 20, 30, 0.95, 40, 50., 60, -0.1, 32u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(gi_aco_mo_evolve_test)
{
    // We check that the problem is checked to be suitable




}

BOOST_AUTO_TEST_CASE(gi_aco_mo_setters_getters_test)
{
    gi_aco_mo user_algo{1u, 0.95, 10, 20, 30, 0.95, 40, 50., 60, 0.95, 32u};
    user_algo.set_verbosity(200u);
    BOOST_CHECK(user_algo.get_verbosity() == 200u);
    user_algo.set_seed(23456u);
    BOOST_CHECK(user_algo.get_seed() == 23456u);
    BOOST_CHECK(user_algo.get_name().find("gi_aco_mo") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Verbosity") != std::string::npos);
    // BOOST_CHECK_NO_THROW(user_algo.get_log());
}

BOOST_AUTO_TEST_CASE(gi_aco_mo_zdt5_test)
{

}

BOOST_AUTO_TEST_CASE(gi_aco_mo_serialization_test)
{


}
