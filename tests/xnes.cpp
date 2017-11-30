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

#define BOOST_TEST_MODULE cmaes_test
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <limits> //  std::numeric_limits<double>::infinity();
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/xnes.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/utils/generic.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(xnes_algorithm_construction)
{
    xnes user_algo{10u};
}

BOOST_AUTO_TEST_CASE(cmaes_evolve_test)
{
    {
        unsigned dim = 2u;
        unsigned popsize = 20u;
        detail::random_engine_type m_e(pagmo::random_device::next());
        // Here we only test that evolution is deterministic if the
        // seed is controlled
        problem prob{rosenbrock{dim}};
        population pop1{prob};
        std::normal_distribution<double> normally_distributed_number(1., 1.);
        std::vector<double> tmp(dim);
        for (int i =0; i< popsize;++i) {
            for (decltype(dim) j = 0u; j < dim; ++j) {
                tmp[j] = 1. + normally_distributed_number(m_e);
            }
            pop1.push_back(tmp);
        }

        xnes user_algo1{400u, -1, -1, -1, 1., 1e-8, 1e-8, false};
        user_algo1.set_verbosity(1u);
        pop1 = user_algo1.evolve(pop1);
        print("Bestx: ", pop1.champion_x());
    }
}