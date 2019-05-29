/* Copyright 2017-2018 PaGMO development team

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

#define BOOST_TEST_MODULE wfg_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <pagmo/problem.hpp>
#include <pagmo/problems/wfg.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(wfg_construction_test)
{
    wfg wfg_default{};
    wfg wfg1{1u, 10u, 5u, 8u};

    BOOST_CHECK_THROW((wfg{10u, 4u, 5u, 2u}), std::invalid_argument);
    BOOST_CHECK_THROW((wfg{8u, 0u, 5u, 2u}), std::invalid_argument);
    BOOST_CHECK_THROW((wfg{8u, 4u, 1u, 1u}), std::invalid_argument);
    BOOST_CHECK_THROW((wfg{8u, 4u, 5u, 5u}), std::invalid_argument);
    BOOST_CHECK_THROW((wfg{8u, 4u, 5u, 0u}), std::invalid_argument);
    BOOST_CHECK_THROW((wfg{8u, 4u, 4u, 2u}), std::invalid_argument);
    BOOST_CHECK_THROW((wfg{2u, 5u, 5u, 2u}), std::invalid_argument);
    BOOST_CHECK_THROW((wfg{3u, 5u, 5u, 2u}), std::invalid_argument);

    BOOST_CHECK_NO_THROW(problem{wfg_default});
    BOOST_CHECK_NO_THROW(problem{wfg1});
    // We also test get_nobj() here as not to add one more small test
    BOOST_CHECK(wfg1.get_nobj() == 5u);
    BOOST_CHECK(wfg_default.get_nobj() == 3u);
    // We also test get_name()
    BOOST_CHECK(wfg1.get_name().find("WFG1") != std::string::npos);
    // And the decision vector dimension
    BOOST_CHECK(problem(wfg1).get_nx() == 10u);
    BOOST_CHECK(problem(wfg_default).get_nx() == 5u);
}
