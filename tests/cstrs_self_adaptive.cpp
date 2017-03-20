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
    population pop{prob, 20u};
    detail::apply_adaptive_penalty dummy{pop};
    print(dummy);
    algorithm algo{de{1u}};
    // algo.set_verbosity(1u);
    population new_pop{dummy, 20u};

    for (int i = 0; i < 1500u; ++i) {

        auto best_idx = pop.best_idx(); // ctol here
        auto best_x = pop.get_x()[best_idx];
        auto best_f = pop.get_f()[best_idx];
        auto worst_idx = pop.worst_idx();

        new_pop.get_problem().extract<detail::apply_adaptive_penalty>()->update();
        for (int i = 0; i < new_pop.size(); ++i) {
            new_pop.set_x(i, pop.get_x()[i]);
        }
        new_pop = algo.evolve(new_pop);
        for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
            auto x = new_pop.get_x()[i];
            auto it_f = new_pop.get_problem().extract<detail::apply_adaptive_penalty>()->m_fitness_map.find(
                new_pop.get_problem().extract<detail::apply_adaptive_penalty>()->m_decision_vector_hash(x));
            if (it_f
                != new_pop.get_problem().extract<detail::apply_adaptive_penalty>()->m_fitness_map.end()) { // cash hit
                pop.set_xf(i, x, it_f->second);
            } else { // we have to compute the fitness (this will increase the feval counter in the ref pop problem )
                pop.set_x(i, x);
            }
        }
        pop.set_xf(worst_idx, best_x, best_f);
    }
    print("\n", pop.champion_f());
    print("\n", pop.get_problem().get_fevals());

    algorithm algo2{cstrs_self_adaptive{de{1u}, 1500u}};
    population pop2{prob, 20u};
    pop2 = algo2.evolve(pop2);
    print("\n", pop2.champion_f());
    print("\n", pop2.get_problem().get_fevals());
}
