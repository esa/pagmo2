/* Copyright 2017-2020 PaGMO development team

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

#define BOOST_TEST_MODULE free_form
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <sstream>

#include <pagmo/s11n.hpp>
#include <pagmo/topologies/free_form.hpp>
#include <pagmo/topology.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(basic_test)
{
    BOOST_CHECK(is_udt<free_form>::value);

    free_form f0;
    BOOST_CHECK(f0.num_vertices() == 0u);

    f0.push_back();
    f0.push_back();
    f0.push_back();

    BOOST_CHECK(f0.num_vertices() == 3u);

    BOOST_CHECK(f0.get_connections(0).first.empty());
    BOOST_CHECK(f0.get_connections(0).second.empty());

    BOOST_CHECK(f0.get_connections(1).first.empty());
    BOOST_CHECK(f0.get_connections(1).second.empty());

    BOOST_CHECK(f0.get_connections(2).first.empty());
    BOOST_CHECK(f0.get_connections(2).second.empty());

    BOOST_CHECK(!f0.are_adjacent(0, 1));
    BOOST_CHECK(!f0.are_adjacent(0, 2));
    BOOST_CHECK(!f0.are_adjacent(1, 2));

    BOOST_CHECK(f0.get_name() == "Free form");
    BOOST_CHECK(topology{f0}.get_name() == "Free form");

    // Minimal serialization test.
    {
        topology t0(f0);
        std::stringstream ss;
        {
            boost::archive::binary_oarchive oarchive(ss);
            oarchive << t0;
        }
        topology t1;
        BOOST_CHECK(!t1.is<free_form>());
        {
            boost::archive::binary_iarchive iarchive(ss);
            iarchive >> t1;
        }
        BOOST_CHECK(t1.is<free_form>());
        BOOST_CHECK(t1.extract<free_form>()->num_vertices() == 3u);

        BOOST_CHECK(t1.extract<free_form>()->get_connections(0).first.empty());
        BOOST_CHECK(t1.extract<free_form>()->get_connections(0).second.empty());

        BOOST_CHECK(t1.extract<free_form>()->get_connections(1).first.empty());
        BOOST_CHECK(t1.extract<free_form>()->get_connections(1).second.empty());

        BOOST_CHECK(t1.extract<free_form>()->get_connections(2).first.empty());
        BOOST_CHECK(t1.extract<free_form>()->get_connections(2).second.empty());
    }

    // Example of cout printing for ring.
    std::cout << topology{f0}.get_extra_info() << '\n';

    // Add a couple of edges.
    f0.add_edge(0, 1, .5);
    f0.add_edge(2, 0);

    BOOST_CHECK(f0.are_adjacent(0, 1));
    BOOST_CHECK(f0.get_connections(1).second[0] == .5);
    BOOST_CHECK(!f0.are_adjacent(1, 0));
    BOOST_CHECK(!f0.are_adjacent(0, 2));
    BOOST_CHECK(f0.are_adjacent(2, 0));
    BOOST_CHECK(f0.get_connections(0).second[0] == 1);
    BOOST_CHECK(!f0.are_adjacent(1, 2));

    std::cout << topology{f0}.get_extra_info() << '\n';
}
