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

#define BOOST_TEST_MODULE base_bgl_topology
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>

#include <pagmo/s11n.hpp>
#include <pagmo/topologies/base_bgl_topology.hpp>

using namespace pagmo;

using bbt = base_bgl_topology;

BOOST_AUTO_TEST_CASE(basic_test)
{
    bbt t0;
    BOOST_CHECK(t0.num_vertices() == 0u);

    t0.add_vertex();
    BOOST_CHECK(t0.num_vertices() == 1u);
    BOOST_CHECK(t0.get_connections(0).first.empty());
    BOOST_CHECK(t0.get_connections(0).second.empty());

    t0.add_vertex();
    t0.add_vertex();
    t0.add_vertex();
    BOOST_CHECK(t0.num_vertices() == 4u);
    BOOST_CHECK(t0.get_connections(0).first.empty());
    BOOST_CHECK(t0.get_connections(0).second.empty());
    BOOST_CHECK(t0.get_connections(1).first.empty());
    BOOST_CHECK(t0.get_connections(1).second.empty());
    BOOST_CHECK(t0.get_connections(2).first.empty());
    BOOST_CHECK(t0.get_connections(2).second.empty());
    BOOST_CHECK(t0.get_connections(3).first.empty());
    BOOST_CHECK(t0.get_connections(3).second.empty());

    t0.add_edge(0, 1);
    t0.add_edge(0, 2);
    BOOST_CHECK(t0.are_adjacent(0, 1));
    BOOST_CHECK(t0.are_adjacent(0, 2));
    BOOST_CHECK(!t0.are_adjacent(1, 0));
    BOOST_CHECK(!t0.are_adjacent(2, 0));

    t0.add_edge(1, 0);
    t0.add_edge(2, 0);
    BOOST_CHECK(t0.get_connections(0).first.size() == 2u);
    BOOST_CHECK(t0.get_connections(0).second.size() == 2u);
    BOOST_CHECK(t0.get_connections(1).first.size() == 1u);
    BOOST_CHECK(t0.get_connections(1).second.size() == 1u);
    BOOST_CHECK(t0.get_connections(2).first.size() == 1u);
    BOOST_CHECK(t0.get_connections(2).second.size() == 1u);

    t0.remove_edge(0, 2);
    BOOST_CHECK(t0.get_connections(2).first.empty());
    BOOST_CHECK(t0.get_connections(2).second.empty());

    t0.set_weight(0, 1, .5);
    BOOST_CHECK(t0.get_connections(1).second.size() == 1u);
    BOOST_CHECK(t0.get_connections(1).second[0] == .5);

    t0.set_all_weights(.25);
    BOOST_CHECK(t0.get_connections(0).second.size() == 2u);
    BOOST_CHECK(t0.get_connections(0).second[0] == .25);
    BOOST_CHECK(t0.get_connections(0).second[1] == .25);

    // Test copy/move.
    auto t1(t0);
    BOOST_CHECK(t1.get_connections(0).second.size() == 2u);
    BOOST_CHECK(t1.get_connections(0).second[0] == .25);
    BOOST_CHECK(t1.get_connections(0).second[1] == .25);

    auto t2(std::move(t1));
    BOOST_CHECK(t2.get_connections(0).second.size() == 2u);
    BOOST_CHECK(t2.get_connections(0).second[0] == .25);
    BOOST_CHECK(t2.get_connections(0).second[1] == .25);

    bbt t3;
    t3 = t2;
    BOOST_CHECK(t3.get_connections(0).second.size() == 2u);
    BOOST_CHECK(t3.get_connections(0).second[0] == .25);
    BOOST_CHECK(t3.get_connections(0).second[1] == .25);

    bbt t4;
    t4 = std::move(t3);
    BOOST_CHECK(t4.get_connections(0).second.size() == 2u);
    BOOST_CHECK(t4.get_connections(0).second[0] == .25);
    BOOST_CHECK(t4.get_connections(0).second[1] == .25);

    const auto str = t4.get_extra_info();
    BOOST_CHECK(boost::contains(str, "Number of vertices: 4"));
    BOOST_CHECK(boost::contains(str, "Number of edges: 3"));
}

BOOST_AUTO_TEST_CASE(error_handling)
{
    bbt t0;

    BOOST_CHECK_EXCEPTION(t0.are_adjacent(0, 1), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "invalid vertex index in a BGL topology: the index is 0, but the number of vertices is only 0");
    });

    t0.add_vertex();
    t0.add_vertex();
    t0.add_vertex();

    BOOST_CHECK_EXCEPTION(t0.get_connections(42), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "invalid vertex index in a BGL topology: the index is 42, but the number of vertices is only 3");
    });

    BOOST_CHECK_EXCEPTION(t0.add_edge(4, 5), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "invalid vertex index in a BGL topology: the index is 4, but the number of vertices is only 3");
    });

    t0.add_edge(0, 2);
    BOOST_CHECK_EXCEPTION(t0.add_edge(0, 2), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(),
                               "cannot add an edge in a BGL topology: there is already an edge connecting 0 to 2");
    });

    t0.remove_edge(0, 2);
    BOOST_CHECK_EXCEPTION(t0.remove_edge(0, 2), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(),
                               "cannot remove an edge in a BGL topology: there is no edge connecting 0 to 2");
    });

    t0.add_edge(0, 2);
    t0.set_weight(0, 2, .2);
    BOOST_CHECK_EXCEPTION(t0.set_weight(0, 2, -1.), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(), " is not in the [0., 1.] range");
    });
    BOOST_CHECK_EXCEPTION(t0.set_weight(0, 2, std::numeric_limits<double>::infinity()), std::invalid_argument,
                          [](const std::invalid_argument &ia) { return boost::contains(ia.what(), " is not finite"); });
}

BOOST_AUTO_TEST_CASE(s11n_test)
{
    bbt t0;
    t0.add_vertex();
    t0.add_vertex();
    t0.add_vertex();
    t0.add_vertex();
    t0.add_edge(0, 1);
    t0.add_edge(0, 2);
    t0.add_edge(1, 0);
    t0.set_weight(0, 1, .5);

    // Minimal serialization test.
    {
        std::stringstream ss;
        {
            boost::archive::binary_oarchive oarchive(ss);
            oarchive << t0;
        }
        bbt t1;
        {
            boost::archive::binary_iarchive iarchive(ss);
            iarchive >> t1;
        }
        BOOST_CHECK(t1.num_vertices() == 4u);
        BOOST_CHECK(t1.are_adjacent(0, 1));
        BOOST_CHECK(t1.are_adjacent(0, 2));
        BOOST_CHECK(t1.are_adjacent(1, 0));
        BOOST_CHECK(!t1.are_adjacent(2, 0));
        BOOST_CHECK(t1.get_connections(1).second.size() == 1u);
        BOOST_CHECK(t1.get_connections(1).second[0] == .5);
    }
}
