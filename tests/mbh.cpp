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
    compass_search inner_algo{100u, 0.1, 0.001, 0.7};
    {
    mbh user_algo{inner_algo, 5, 1e-3};
    BOOST_CHECK((user_algo.get_perturb() == vector_double{1e-3}));
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK((user_algo.get_log() == mbh::log_type{}));
    }
    {
    mbh user_algo{inner_algo, 5, {1e-3, 1e-2, 1e-3, 1e-2}};
    BOOST_CHECK((user_algo.get_perturb() == vector_double{1e-3, 1e-2,1e-3, 1e-2}));
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK((user_algo.get_log() == mbh::log_type{}));
    algorithm algo{user_algo};
    population pop{problem{hock_schittkowsky_71{}}, 1u};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);
    }
}
