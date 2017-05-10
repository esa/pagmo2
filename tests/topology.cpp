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

#include <pagmo/topology.hpp>

#define BOOST_TEST_MODULE problem_test
#include <boost/test/included/unit_test.hpp>

#include <string>

using namespace pagmo;

struct unconnected2 : unconnected {
    std::string get_extra_info() const
    {
        return "foobar";
    }
};

BOOST_AUTO_TEST_CASE(topology_construction_test)
{
    topology t;
    BOOST_CHECK(t.get_name() == "Unconnected");
    BOOST_CHECK(t.get_extra_info() == "");
    BOOST_CHECK(t.is<unconnected>());
    BOOST_CHECK(t.extract<unconnected>() != nullptr);
    BOOST_CHECK(static_cast<const topology &>(t).extract<unconnected>() != nullptr);
    BOOST_CHECK(!t.is<unconnected2>());
    BOOST_CHECK(t.extract<unconnected2>() == nullptr);
    BOOST_CHECK(static_cast<const topology &>(t).extract<unconnected2>() == nullptr);
}
