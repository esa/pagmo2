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

#define BOOST_TEST_MODULE cec2009_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <exception>
#include <iostream>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/cec2009.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

using namespace pagmo;
using statics = detail::cec2009_statics<>;

BOOST_AUTO_TEST_CASE(cec2009_construction_test)
{
    // We check that all problems can be constructed
    for (unsigned i = 1u; i <= 10u; ++i) {
        cec2009 udp{i, false};
        print(udp.fitness(vector_double(30, 0.5)), "\n");
    }
    for (unsigned i = 1u; i <= 10u; ++i) {
        cec2009 udp{i, true};
        print(udp.fitness(vector_double(30, 0.5)), "\n");
    }
    // We check that wrong problem ids and dimensions cannot be constructed
    // BOOST_CHECK_THROW((cec2006{0u}), std::invalid_argument);
    // BOOST_CHECK_THROW((cec2006{29u}), std::invalid_argument);
}
