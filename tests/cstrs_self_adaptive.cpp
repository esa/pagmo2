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
    population pop{cec2006{}, 20u};
    detail::apply_adaptive_penalty dummy{pop};
    print(dummy);
    algorithm algo{de{500u}};
    algo.set_verbosity(1u);
    population new_pop{problem{dummy}, 20u};

    for (int i = 0; i < 1u; ++i) {
        new_pop = algo.evolve(new_pop);
        for (int i = 0; i < pop.size(); ++i) {
            pop.set_x(i, new_pop.get_x()[i]);
        }
        new_pop.get_problem().extract<detail::apply_adaptive_penalty>()->update();
        print("\n", problem{cec2006{}}.fitness(new_pop.champion_x()));
        print("\n", dummy.fitness(new_pop.get_x()[new_pop.best_idx()]));
    }
}
