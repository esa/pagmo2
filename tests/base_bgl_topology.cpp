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

#define BOOST_TEST_MODULE base_bgl_topology
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <atomic>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/graph/adjacency_list.hpp>

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
    BOOST_CHECK(t0.get_edge_weight(1, 0) == 1.);
    BOOST_CHECK(t0.get_edge_weight(2, 0) == 1.);

    auto conns = t0.get_connections(0);
    using c_vec = decltype(conns.first);
    using w_vec = decltype(conns.second);

    BOOST_CHECK((conns.first == c_vec{1, 2} || conns.first == c_vec{2, 1}));
    BOOST_CHECK((conns.second == w_vec{1., 1.}));

    conns = t0.get_connections(1);

    BOOST_CHECK((conns.first == c_vec{0}));
    BOOST_CHECK((conns.second == w_vec{1.}));

    conns = t0.get_connections(2);

    BOOST_CHECK((conns.first == c_vec{0}));
    BOOST_CHECK((conns.second == w_vec{1.}));

    t0.remove_edge(0, 2);
    BOOST_CHECK(t0.get_connections(2).first.empty());
    BOOST_CHECK(t0.get_connections(2).second.empty());

    t0.set_weight(0, 1, .5);
    BOOST_CHECK(t0.get_edge_weight(0, 1) == .5);

    conns = t0.get_connections(1);

    BOOST_CHECK((conns.first == c_vec{0}));
    BOOST_CHECK((conns.second == w_vec{.5}));

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
    BOOST_CHECK_EXCEPTION(t0.set_all_weights(-1.), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(), " is not in the [0., 1.] range");
    });
    BOOST_CHECK_EXCEPTION(t0.set_weight(0, 2, std::numeric_limits<double>::infinity()), std::invalid_argument,
                          [](const std::invalid_argument &ia) { return boost::contains(ia.what(), " is not finite"); });
    BOOST_CHECK_EXCEPTION(t0.set_all_weights(std::numeric_limits<double>::infinity()), std::invalid_argument,
                          [](const std::invalid_argument &ia) { return boost::contains(ia.what(), " is not finite"); });
    BOOST_CHECK_EXCEPTION(t0.set_weight(0, 1, .2), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "cannot set the weight of an edge in a BGL topology: the vertex 0 is not connected to vertex 1");
    });

    BOOST_CHECK_EXCEPTION(t0.get_edge_weight(0, 1), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "cannot get the weight of an edge in a BGL topology: the vertex 0 is not connected to vertex 1");
    });
    BOOST_CHECK_EXCEPTION(t0.get_edge_weight(0, 10), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "invalid vertex index in a BGL topology: the index is 10, but the number of vertices is only 3");
    });
    BOOST_CHECK_EXCEPTION(t0.get_edge_weight(11, 10), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "invalid vertex index in a BGL topology: the index is 11, but the number of vertices is only 3");
    });
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

BOOST_AUTO_TEST_CASE(thread_torture_test)
{
    std::atomic<int> barrier(0), failures(0);

    bbt t0;
    t0.add_vertex();
    t0.add_vertex();
    t0.add_vertex();
    t0.add_vertex();
    t0.add_edge(0, 1);
    t0.add_edge(0, 2);
    t0.add_edge(1, 0);
    t0.set_weight(0, 1, .5);

    std::vector<std::thread> threads;
    for (auto i = 0; i < 10; ++i) {
        threads.emplace_back([&barrier, &failures, &t0]() {
            ++barrier;
            while (barrier.load() != 10) {
            }

            for (int j = 0; j < 100; ++j) {
                auto t1(t0);
                bbt t2;
                t2 = t0;

                t0.add_vertex();
                failures += t0.num_vertices() < 4u;
                t0.add_vertex();
                failures += !t0.are_adjacent(0, 1);
                t0.add_vertex();
                failures += t0.get_connections(0).first.size() == 0u;
                t0.add_vertex();

                try {
                    t0.add_edge(0u, 4u);
                    t0.set_weight(0u, 4u, .3);
                    t0.remove_edge(0u, 4u);
                } catch (const std::invalid_argument &) {
                }

                t0.set_all_weights(.1);

                failures += t0.get_extra_info().empty();

                t0 = std::move(t2);
            }
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    BOOST_CHECK(failures.load() == 0);
}

BOOST_AUTO_TEST_CASE(to_bgl_test)
{
    bbt t0;

    auto b = t0.to_bgl();

    BOOST_CHECK(boost::num_vertices(b) == 0u);

    t0.add_vertex();
    t0.add_vertex();
    t0.add_vertex();

    b = t0.to_bgl();
    BOOST_CHECK(boost::num_vertices(b) == 3u);
    auto a_vertices = boost::adjacent_vertices(boost::vertex(0, b), b);
    BOOST_CHECK(a_vertices.first == a_vertices.second);
    a_vertices = boost::adjacent_vertices(boost::vertex(1, b), b);
    BOOST_CHECK(a_vertices.first == a_vertices.second);
    a_vertices = boost::adjacent_vertices(boost::vertex(2, b), b);
    BOOST_CHECK(a_vertices.first == a_vertices.second);

    t0.add_edge(0, 1, .25);
    t0.add_edge(1, 2, 1);
    b = t0.to_bgl();
    BOOST_CHECK(boost::num_vertices(b) == 3u);
    a_vertices = boost::adjacent_vertices(boost::vertex(0, b), b);
    BOOST_CHECK(a_vertices.second - a_vertices.first == 1);
    auto vi = boost::vertex(0, b);
    for (auto av = boost::adjacent_vertices(vi, b); av.first != av.second; ++av.first) {
        const auto e = boost::edge(vi, boost::vertex(*av.first, b), b);
        BOOST_CHECK(e.second);
        BOOST_CHECK(b[e.first] == .25);
    }
    a_vertices = boost::adjacent_vertices(boost::vertex(1, b), b);
    BOOST_CHECK(a_vertices.second - a_vertices.first == 1);
    vi = boost::vertex(1, b);
    for (auto av = boost::adjacent_vertices(vi, b); av.first != av.second; ++av.first) {
        const auto e = boost::edge(vi, boost::vertex(*av.first, b), b);
        BOOST_CHECK(e.second);
        BOOST_CHECK(b[e.first] == 1.);
    }
    a_vertices = boost::adjacent_vertices(boost::vertex(2, b), b);
    BOOST_CHECK(a_vertices.first == a_vertices.second);
}
