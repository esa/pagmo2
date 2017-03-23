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

#define BOOST_TEST_MODULE archipelago_test
#include <boost/test/included/unit_test.hpp>

#include <utility>

#include <pagmo/algorithms/de.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/rosenbrock.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(archipelago_construction)
{
    using size_type = archipelago::size_type;
    archipelago archi;
    BOOST_CHECK(archi.size() == 0u);
    archipelago archi2(0u, de{}, rosenbrock{}, 10u);
    BOOST_CHECK(archi2.size() == 0u);
    archipelago archi3(5u, de{}, rosenbrock{}, 10u);
    BOOST_CHECK(archi3.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(!archi3[i].busy());
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
    }
    archi3 = archipelago{5u, thread_island{}, de{}, rosenbrock{}, 10u};
    BOOST_CHECK(archi3.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(!archi3[i].busy());
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
    }
    archi3 = archipelago{5u, thread_island{}, de{}, population{rosenbrock{}, 10u}};
    BOOST_CHECK(archi3.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(!archi3[i].busy());
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
    }
    archi3 = archipelago{5u, thread_island{}, de{}, population{rosenbrock{}, 10u, 123u}};
    BOOST_CHECK(archi3.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(!archi3[i].busy());
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
    }
    auto archi4 = archi3;
    BOOST_CHECK(archi4.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(!archi4[i].busy());
        BOOST_CHECK(archi4[i].get_algorithm().is<de>());
        BOOST_CHECK(archi4[i].get_population().size() == 10u);
        BOOST_CHECK(archi4[i].get_population().get_problem().is<rosenbrock>());
    }
    archi4.evolve(10);
    auto archi5 = archi4;
    BOOST_CHECK(archi5.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(!archi5[i].busy());
        BOOST_CHECK(archi5[i].get_algorithm().is<de>());
        BOOST_CHECK(archi5[i].get_population().size() == 10u);
        BOOST_CHECK(archi5[i].get_population().get_problem().is<rosenbrock>());
    }
    archi4.get();
    archi4.evolve(10);
    auto archi6(std::move(archi4));
    BOOST_CHECK(archi6.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(!archi6[i].busy());
        BOOST_CHECK(archi6[i].get_algorithm().is<de>());
        BOOST_CHECK(archi6[i].get_population().size() == 10u);
        BOOST_CHECK(archi6[i].get_population().get_problem().is<rosenbrock>());
    }
    BOOST_CHECK(archi4.size() == 0u);
    archi4 = archi5;
    BOOST_CHECK(archi4.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(!archi4[i].busy());
        BOOST_CHECK(archi4[i].get_algorithm().is<de>());
        BOOST_CHECK(archi4[i].get_population().size() == 10u);
        BOOST_CHECK(archi4[i].get_population().get_problem().is<rosenbrock>());
    }
    archi4 = std::move(archi5);
    BOOST_CHECK(archi4.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(!archi4[i].busy());
        BOOST_CHECK(archi4[i].get_algorithm().is<de>());
        BOOST_CHECK(archi4[i].get_population().size() == 10u);
        BOOST_CHECK(archi4[i].get_population().get_problem().is<rosenbrock>());
    }
    BOOST_CHECK(archi5.size() == 0u);
    // Self assignment.
    archi4 = archi4;
    BOOST_CHECK(archi4.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(!archi4[i].busy());
        BOOST_CHECK(archi4[i].get_algorithm().is<de>());
        BOOST_CHECK(archi4[i].get_population().size() == 10u);
        BOOST_CHECK(archi4[i].get_population().get_problem().is<rosenbrock>());
    }
#if !defined(__clang__)
    archi4 = std::move(archi4);
    BOOST_CHECK(archi4.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(!archi4[i].busy());
        BOOST_CHECK(archi4[i].get_algorithm().is<de>());
        BOOST_CHECK(archi4[i].get_population().size() == 10u);
        BOOST_CHECK(archi4[i].get_population().get_problem().is<rosenbrock>());
    }
#endif
}
