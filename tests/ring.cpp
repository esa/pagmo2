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

#define BOOST_TEST_MODULE ring
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <pagmo/s11n.hpp>
#include <pagmo/topologies/ring.hpp>
#include <pagmo/topology.hpp>

using namespace pagmo;

void verify_ring_topology(const ring &r)
{
    const auto s = r.num_vertices();

    if (s < 2u) {
        return;
    }

    const auto w = r.get_weight();

    for (std::size_t i = 0; i < s; ++i) {
        const auto conn = r.get_connections(i);
        if (s > 2u) {
            BOOST_CHECK(conn.first.size() == 2u);
            BOOST_CHECK(conn.second.size() == 2u);
        } else {
            BOOST_CHECK(conn.first.size() == 1u);
            BOOST_CHECK(conn.second.size() == 1u);
        }
        BOOST_CHECK(std::all_of(conn.second.begin(), conn.second.end(), [w](double x) { return x == w; }));

        const auto next = (i + 1u) % s;
        const auto prev = (i == 0u) ? (s - 1u) : (i - 1u);
        BOOST_CHECK(std::find(conn.first.begin(), conn.first.end(), next) != conn.first.end());
        BOOST_CHECK(std::find(conn.first.begin(), conn.first.end(), prev) != conn.first.end());
    }
}

BOOST_AUTO_TEST_CASE(basic_test)
{
    ring r0;
    BOOST_CHECK(r0.get_weight() == 1);
    BOOST_CHECK(r0.num_vertices() == 0u);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 1u);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 2u);
    verify_ring_topology(r0);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 3u);
    verify_ring_topology(r0);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 4u);
    verify_ring_topology(r0);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 5u);
    verify_ring_topology(r0);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 6u);
    verify_ring_topology(r0);

    r0 = ring(.5);
    BOOST_CHECK(r0.get_weight() == .5);
    BOOST_CHECK(r0.num_vertices() == 0u);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 1u);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 2u);
    verify_ring_topology(r0);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 3u);
    verify_ring_topology(r0);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 4u);
    verify_ring_topology(r0);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 5u);
    verify_ring_topology(r0);

    r0.push_back();
    BOOST_CHECK(r0.num_vertices() == 6u);
    verify_ring_topology(r0);

    BOOST_CHECK(r0.get_name() == "Ring");

    // Minimal serialization test.
    {
        topology t0(r0);
        std::stringstream ss;
        {
            boost::archive::binary_oarchive oarchive(ss);
            oarchive << t0;
        }
        topology t1;
        BOOST_CHECK(!t1.is<ring>());
        {
            boost::archive::binary_iarchive iarchive(ss);
            iarchive >> t1;
        }
        BOOST_CHECK(t1.is<ring>());
        BOOST_CHECK(t1.extract<ring>()->num_vertices() == 6u);
        BOOST_CHECK(t1.extract<ring>()->get_weight() == .5);
        verify_ring_topology(*t1.extract<ring>());
    }

    // Ctor from edge weight.
    BOOST_CHECK_EXCEPTION(r0 = ring(-2), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(), " is not in the [0., 1.] range");
    });

    // Ctor from number of vertices and edge weight.
    BOOST_CHECK_EXCEPTION(r0 = ring(0, -2), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(), " is not in the [0., 1.] range");
    });

    r0 = ring(0, 0);

    BOOST_CHECK(r0.get_weight() == 0.);
    BOOST_CHECK(r0.num_vertices() == 0u);
    verify_ring_topology(r0);

    r0 = ring(1, .2);
    BOOST_CHECK(r0.get_weight() == .2);
    BOOST_CHECK(r0.num_vertices() == 1u);
    verify_ring_topology(r0);

    r0 = ring(2, .3);
    BOOST_CHECK(r0.get_weight() == .3);
    BOOST_CHECK(r0.num_vertices() == 2u);
    verify_ring_topology(r0);

    r0 = ring(3, .4);
    BOOST_CHECK(r0.get_weight() == .4);
    BOOST_CHECK(r0.num_vertices() == 3u);
    verify_ring_topology(r0);

    r0 = ring(4, .5);
    BOOST_CHECK(r0.get_weight() == .5);
    BOOST_CHECK(r0.num_vertices() == 4u);
    verify_ring_topology(r0);

    // Example of cout printing for ring.
    r0.push_back();
    r0.push_back();
    r0.push_back();
    r0.push_back();

    r0.set_weight(0, 1, .1);
    r0.set_weight(4, 5, .7);

    std::cout << r0.get_extra_info() << '\n';
}

BOOST_AUTO_TEST_CASE(to_bgl_test)
{
    BOOST_CHECK(has_to_bgl<ring>::value);

    auto r0 = ring(4, .5);
    BOOST_CHECK(boost::num_vertices(r0.to_bgl()) == 4);
    BOOST_CHECK(boost::num_edges(r0.to_bgl()) == 8);
}
