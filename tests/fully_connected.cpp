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

#define BOOST_TEST_MODULE fully_connected
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <pagmo/s11n.hpp>
#include <pagmo/topologies/fully_connected.hpp>
#include <pagmo/topology.hpp>

using namespace pagmo;

void verify_fully_connected_topology(const fully_connected &f)
{
    const auto s = f.num_vertices();

    if (!s) {
        BOOST_CHECK_EXCEPTION(f.get_connections(0), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Cannot get the connections to the vertex at index 0 in a fully "
                                              "connected topology: the number of vertices in the topology is only 0");
        });

        return;
    }

    if (s == 1u) {
        BOOST_CHECK(f.get_connections(0).first.empty());
        BOOST_CHECK(f.get_connections(0).second.empty());

        BOOST_CHECK_EXCEPTION(f.get_connections(1), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Cannot get the connections to the vertex at index 1 in a fully "
                                              "connected topology: the number of vertices in the topology is only 1");
        });

        return;
    }

    const auto w = f.get_weight();

    for (std::size_t i = 0; i < s; ++i) {
        const auto conns = f.get_connections(i);

        BOOST_CHECK(conns.first.size() == s - 1u);
        BOOST_CHECK(conns.second.size() == s - 1u);

        BOOST_CHECK(std::all_of(conns.second.begin(), conns.second.end(), [w](double x) { return x == w; }));

        for (std::size_t j = 0; j < i; ++j) {
            BOOST_CHECK(std::find(conns.first.begin(), conns.first.end(), j) != conns.first.end());
        }
        for (std::size_t j = i + 1u; j < s; ++j) {
            BOOST_CHECK(std::find(conns.first.begin(), conns.first.end(), j) != conns.first.end());
        }
    }
}

BOOST_AUTO_TEST_CASE(basic_test)
{
    {
        // Default construct, push back a few times.
        fully_connected r0;
        BOOST_CHECK(r0.get_weight() == 1);
        BOOST_CHECK(r0.num_vertices() == 0u);
        verify_fully_connected_topology(r0);

        r0.push_back();
        BOOST_CHECK(r0.num_vertices() == 1u);
        verify_fully_connected_topology(r0);

        r0.push_back();
        BOOST_CHECK(r0.num_vertices() == 2u);
        verify_fully_connected_topology(r0);

        r0.push_back();
        BOOST_CHECK(r0.num_vertices() == 3u);
        verify_fully_connected_topology(r0);

        r0.push_back();
        r0.push_back();
        r0.push_back();
        r0.push_back();
        BOOST_CHECK(r0.num_vertices() == 7u);
        verify_fully_connected_topology(r0);
    }

    {
        // Ctor from weight.
        fully_connected r0(.2);
        BOOST_CHECK(r0.get_weight() == .2);
        BOOST_CHECK(r0.num_vertices() == 0u);
        verify_fully_connected_topology(r0);

        r0.push_back();
        BOOST_CHECK(r0.num_vertices() == 1u);
        verify_fully_connected_topology(r0);

        r0.push_back();
        BOOST_CHECK(r0.num_vertices() == 2u);
        verify_fully_connected_topology(r0);

        r0.push_back();
        BOOST_CHECK(r0.num_vertices() == 3u);
        verify_fully_connected_topology(r0);

        r0.push_back();
        r0.push_back();
        r0.push_back();
        r0.push_back();
        BOOST_CHECK(r0.num_vertices() == 7u);
        verify_fully_connected_topology(r0);
    }

    {
        // Ctor from nedges and weight.
        fully_connected r0(0, .2);
        BOOST_CHECK(r0.get_weight() == .2);
        BOOST_CHECK(r0.num_vertices() == 0u);
        verify_fully_connected_topology(r0);
    }

    {
        // Ctor from nedges and weight.
        fully_connected r0(7, .2);
        BOOST_CHECK(r0.get_weight() == .2);
        BOOST_CHECK(r0.num_vertices() == 7u);
        verify_fully_connected_topology(r0);
    }

    {
        // Copy/move ctors.
        fully_connected r0(7, .2), r1(r0), r2(std::move(r0));

        BOOST_CHECK(r1.get_weight() == .2);
        BOOST_CHECK(r1.num_vertices() == 7u);

        BOOST_CHECK(r1.get_weight() == .2);
        BOOST_CHECK(r2.num_vertices() == 7u);

        verify_fully_connected_topology(r1);
        verify_fully_connected_topology(r2);
    }

    {
        // Name/extra info.
        fully_connected r0(7, .2);

        BOOST_CHECK(r0.get_name() == "Fully connected");
        BOOST_CHECK(boost::contains(r0.get_extra_info(), "Edges' weight:"));

        std::cout << r0.get_extra_info() << '\n';
    }

    // Minimal serialization test.
    fully_connected r0(7, .2);
    {
        topology t0(r0);
        std::stringstream ss;
        {
            boost::archive::binary_oarchive oarchive(ss);
            oarchive << t0;
        }
        topology t1;
        BOOST_CHECK(!t1.is<fully_connected>());
        {
            boost::archive::binary_iarchive iarchive(ss);
            iarchive >> t1;
        }
        BOOST_CHECK(t1.is<fully_connected>());
        BOOST_CHECK(t1.extract<fully_connected>()->num_vertices() == 7u);
        BOOST_CHECK(t1.extract<fully_connected>()->get_weight() == .2);
        verify_fully_connected_topology(*t1.extract<fully_connected>());
    }
}

BOOST_AUTO_TEST_CASE(to_bgl_test)
{
    BOOST_CHECK(has_to_bgl<fully_connected>::value);

    auto g0 = fully_connected{}.to_bgl();
    BOOST_CHECK(boost::num_vertices(g0) == 0u);

    g0 = fully_connected{1, .5}.to_bgl();
    BOOST_CHECK(boost::num_vertices(g0) == 1u);
    auto vi = boost::vertex(0, g0);
    auto av = boost::adjacent_vertices(vi, g0);
    BOOST_CHECK(av.first == av.second);

    g0 = fully_connected{2, .5}.to_bgl();
    BOOST_CHECK(boost::num_vertices(g0) == 2u);

    vi = boost::vertex(0, g0);
    av = boost::adjacent_vertices(vi, g0);
    auto e = boost::edge(boost::vertex(*av.first, g0), vi, g0);
    BOOST_CHECK(e.second);
    BOOST_CHECK(*av.first == 1);
    BOOST_CHECK(g0[e.first] == .5);
    BOOST_CHECK(++av.first == av.second);

    vi = boost::vertex(1, g0);
    av = boost::adjacent_vertices(vi, g0);
    e = boost::edge(boost::vertex(*av.first, g0), vi, g0);
    BOOST_CHECK(e.second);
    BOOST_CHECK(*av.first == 0);
    BOOST_CHECK(g0[e.first] == .5);
    BOOST_CHECK(++av.first == av.second);

    g0 = fully_connected{3, .5}.to_bgl();
    BOOST_CHECK(boost::num_vertices(g0) == 3u);

    vi = boost::vertex(0, g0);
    av = boost::adjacent_vertices(vi, g0);
    e = boost::edge(boost::vertex(*av.first, g0), vi, g0);
    BOOST_CHECK(e.second);
    BOOST_CHECK(*av.first == 1 || *av.first == 2);
    BOOST_CHECK(g0[e.first] == .5);
    ++av.first;
    e = boost::edge(boost::vertex(*av.first, g0), vi, g0);
    BOOST_CHECK(e.second);
    BOOST_CHECK(*av.first == 1 || *av.first == 2);
    BOOST_CHECK(g0[e.first] == .5);
    BOOST_CHECK(++av.first == av.second);

    vi = boost::vertex(1, g0);
    av = boost::adjacent_vertices(vi, g0);
    e = boost::edge(boost::vertex(*av.first, g0), vi, g0);
    BOOST_CHECK(e.second);
    BOOST_CHECK(*av.first == 0 || *av.first == 2);
    BOOST_CHECK(g0[e.first] == .5);
    ++av.first;
    e = boost::edge(boost::vertex(*av.first, g0), vi, g0);
    BOOST_CHECK(e.second);
    BOOST_CHECK(*av.first == 0 || *av.first == 2);
    BOOST_CHECK(g0[e.first] == .5);
    BOOST_CHECK(++av.first == av.second);

    vi = boost::vertex(2, g0);
    av = boost::adjacent_vertices(vi, g0);
    e = boost::edge(boost::vertex(*av.first, g0), vi, g0);
    BOOST_CHECK(e.second);
    BOOST_CHECK(*av.first == 0 || *av.first == 1);
    BOOST_CHECK(g0[e.first] == .5);
    ++av.first;
    e = boost::edge(boost::vertex(*av.first, g0), vi, g0);
    BOOST_CHECK(e.second);
    BOOST_CHECK(*av.first == 0 || *av.first == 1);
    BOOST_CHECK(g0[e.first] == .5);
    BOOST_CHECK(++av.first == av.second);
}
