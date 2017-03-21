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

#define BOOST_TEST_MODULE cstrs_self_adaptive_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <cmath>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/cstrs_self_adaptive.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/cec2006.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(cstrs_self_adaptive_construction_test)
{
    problem prob{cec2006{1u}};
    auto c_tol = vector_double(prob.get_nc(), 0.);
    prob.set_c_tol(c_tol);
    algorithm algo2{cstrs_self_adaptive(1500u)};
    population pop2{prob, 20u};
    algo2.set_verbosity(10u);
    pop2 = algo2.evolve(pop2);
    print("\n", pop2.champion_f());
    print("\n", pop2.get_problem().get_fevals());
}
