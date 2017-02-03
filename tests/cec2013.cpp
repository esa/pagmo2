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

#define BOOST_TEST_MODULE cec2013_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <exception>
#include <iostream>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/cec2013.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(cec2013_test)
{
    // We check that all problems can be constructed at all dimensions
    std::vector<unsigned int> allowed_dims = {2u, 5u, 10u, 20u, 30u, 40u, 50u, 60u, 70u, 80u, 90u, 100u};
    for (unsigned int i=1u ; i<=18u; ++i) {
        for (auto dim : allowed_dims) {
            BOOST_CHECK_NO_THROW((cec2013{i, dim, "cec2013_data/"}));
        }
    }
    // We check that wrong problem ids and dimensions cannot be constructed
    BOOST_CHECK_THROW((cec2013{0u, 2u, "cec2013_data/"}), std::invalid_argument);
    BOOST_CHECK_THROW((cec2013{19u, 2u, "cec2013_data/"}), std::invalid_argument);
    BOOST_CHECK_THROW((cec2013{10u, 3u, "cec2013_data/"}), std::invalid_argument);
    BOOST_CHECK_THROW((cec2013{10u, 3u, "cec2013_data_mispelled/"}), std::invalid_argument);
    BOOST_CHECK_THROW((cec2013{10u, 3u, "cec2013_data/only_shift/"}), std::invalid_argument);
}
