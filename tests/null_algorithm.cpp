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

#define BOOST_TEST_MODULE null_algo_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/algorithms/null_algorithm.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(algorithm_construction_and_evolve)
{
    // Trivial checks
    null_algorithm user_algo{};
    BOOST_CHECK(user_algo.get_a() == 42.1);
    BOOST_CHECK(user_algo.get_extra_info().find("Useless parameter") != std::string::npos);
    BOOST_CHECK(user_algo.get_name().find("Null") != std::string::npos);
    // Evolve check (population does not change)
    rosenbrock user_prob{};
    population pop(user_prob, 20u);
    auto evolved_pop = user_algo.evolve(pop);
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        BOOST_CHECK(pop.get_x()[i] == evolved_pop.get_x()[i]);
        BOOST_CHECK(pop.get_f()[i] == evolved_pop.get_f()[i]);
        BOOST_CHECK(pop.get_ID()[i] == evolved_pop.get_ID()[i]);
    }
}

BOOST_AUTO_TEST_CASE(serialization_test)
{
    algorithm algo{null_algorithm{}};
    auto a = algo.extract<null_algorithm>()->get_a();
    // Now serialize, deserialize and compare the result.
    std::stringstream ss;
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(algo);
    }
    // Change the content of p before deserializing.
    algo = algorithm{de{}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(algo);
    }
    auto after_a = algo.extract<null_algorithm>()->get_a();
    BOOST_CHECK_EQUAL(a, after_a);
}
