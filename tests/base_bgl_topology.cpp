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

#include <pagmo/topologies/base_bgl_topology.hpp>

#define BOOST_TEST_MODULE base_bgl_topology_test
#include <boost/test/included/unit_test.hpp>

using namespace pagmo;

using bb_t = base_bgl_topology;

BOOST_AUTO_TEST_CASE(base_bgl_topology_construction_test)
{
    bb_t b;
    b.add_vertex();
    b.add_vertex();
    BOOST_CHECK(b.num_vertices() == 2u);
    BOOST_CHECK(!b.are_adjacent(0, 1));
    BOOST_CHECK(!b.are_adjacent(1, 0));
}
