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

#define BOOST_TEST_MODULE dtlz_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <exception>
#include <iostream>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(zdt_construction_test)
{
    dtlz dtlz_default{};
    dtlz dtlz5{5u, 7u, 3u, 100u};

    BOOST_CHECK_THROW((dtlz{0u, 7u, 3u, 100u}), std::invalid_argument);
    BOOST_CHECK_THROW((dtlz{9u, 7u, 3u, 100u}), std::invalid_argument);
    BOOST_CHECK_THROW((dtlz{1u, 7u, 1u, 100u}), std::invalid_argument);
    BOOST_CHECK_THROW((dtlz{1u, 7u, std::numeric_limits<vector_double::size_type>::max() - 1u, 100u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((dtlz{1u, std::numeric_limits<vector_double::size_type>::max() - 1u, 3u, 100u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((dtlz{1u, 3u, 3u, 100u}), std::invalid_argument);

    BOOST_CHECK_NO_THROW(problem{dtlz_default});
    BOOST_CHECK_NO_THROW(problem{dtlz5});
    // We also test get_nobj() here as not to add one more small test
    BOOST_CHECK(dtlz_default.get_nobj() == 3u);
    // We also test get_name()
    BOOST_CHECK(dtlz5.get_name().find("DTLZ5") != std::string::npos);

    BOOST_CHECK(problem{dtlz5}.get_nx() == 7u);
    BOOST_CHECK(dtlz5.get_bounds().first.size() == 7u);
}

BOOST_AUTO_TEST_CASE(dtlz1_fitness_test)
{
    vector_double dv1{0.5, 0.5, 0.5, 0.5, 0.5};
    vector_double dv2{0.1, 0.2, 0.3, 0.4, 0.5};
    vector_double f1, f2;
    // dtlz1
    dtlz udp{1u, 5u, 3u};
    f1 = {0.125, 0.125, 0.25};
    f2 = {0.059999999999999824, 0.17999999999999947, 2.099999999999994};
    for (unsigned int i = 0u; i < 3; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv1)[i], f1[i], 1e-12);
    }
    for (unsigned int i = 0u; i < 3; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv2)[i], f2[i], 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(dtlz2_fitness_test)
{
    vector_double dv1{0.5, 0.5, 0.5, 0.5, 0.5};
    vector_double dv2{0.1, 0.2, 0.3, 0.4, 0.5};
    vector_double f1, f2;
    // dtlz1
    dtlz udp{2u, 5u, 3u};
    f1 = {0.5000000000000001, 0.5, 0.7071067811865475};
    f2 = {0.9863148040113404, 0.3204731065093832, 0.16425618829224242};
    for (unsigned int i = 0u; i < 3; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv1)[i], f1[i], 1e-12);
    }
    for (unsigned int i = 0u; i < 3; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv2)[i], f2[i], 1e-12);
    }
}
