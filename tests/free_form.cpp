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

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <pagmo/s11n.hpp>
#include <pagmo/topologies/free_form.hpp>
#include <pagmo/topologies/ring.hpp>
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

    // Copy ctor.
    auto f1(f0);

    BOOST_CHECK(f1.num_vertices() == 3u);

    BOOST_CHECK(f1.get_connections(0).first.empty());
    BOOST_CHECK(f1.get_connections(0).second.empty());

    BOOST_CHECK(f1.get_connections(1).first.empty());
    BOOST_CHECK(f1.get_connections(1).second.empty());

    BOOST_CHECK(f1.get_connections(2).first.empty());
    BOOST_CHECK(f1.get_connections(2).second.empty());

    BOOST_CHECK(!f1.are_adjacent(0, 1));
    BOOST_CHECK(!f1.are_adjacent(0, 2));
    BOOST_CHECK(!f1.are_adjacent(1, 2));

    // Move ctor.
    auto f2(std::move(f1));

    BOOST_CHECK(f2.num_vertices() == 3u);

    BOOST_CHECK(f2.get_connections(0).first.empty());
    BOOST_CHECK(f2.get_connections(0).second.empty());

    BOOST_CHECK(f2.get_connections(1).first.empty());
    BOOST_CHECK(f2.get_connections(1).second.empty());

    BOOST_CHECK(f2.get_connections(2).first.empty());
    BOOST_CHECK(f2.get_connections(2).second.empty());

    BOOST_CHECK(!f2.are_adjacent(0, 1));
    BOOST_CHECK(!f2.are_adjacent(0, 2));
    BOOST_CHECK(!f2.are_adjacent(1, 2));

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

    // Example of cout printing for free_form.
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

BOOST_AUTO_TEST_CASE(bgl_ctor)
{
    ring r0{100, .25};
    free_form f0{r0.to_bgl()};

    BOOST_CHECK(f0.num_vertices() == 100u);
    BOOST_CHECK(f0.are_adjacent(0, 1));
    BOOST_CHECK(f0.are_adjacent(1, 0));
    BOOST_CHECK(f0.are_adjacent(0, 99));
    BOOST_CHECK(f0.are_adjacent(99, 0));
    BOOST_CHECK(!f0.are_adjacent(0, 2));
    BOOST_CHECK(!f0.are_adjacent(2, 0));

    for (auto i = 0u; i < 100u; ++i) {
        auto c = f0.get_connections(i);

        BOOST_CHECK(c.first.size() == 2u);
        BOOST_CHECK(c.second.size() == 2u);
        BOOST_CHECK(std::all_of(c.second.begin(), c.second.end(), [](double w) { return w == .25; }));
    }

    // Test error throwing with invalid weights.
    bgl_graph_t bogus;

    boost::add_vertex(bogus);
    boost::add_vertex(bogus);
    boost::add_vertex(bogus);

    auto res = boost::add_edge(boost::vertex(0, bogus), boost::vertex(1, bogus), bogus);
    bogus[res.first] = 0.;
    res = boost::add_edge(boost::vertex(1, bogus), boost::vertex(2, bogus), bogus);
    bogus[res.first] = 2.;

    auto trigger = [&bogus]() { free_form fobus(bogus); };

    BOOST_CHECK_EXCEPTION(trigger(), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "In the constructor of a free_form topology from a graph object, an invalid edge weight of "
                           + std::to_string(2.) + " was detected (the weight must be in the [0., 1.] range)");
    });

    bogus[res.first] = -1.;

    BOOST_CHECK_EXCEPTION(trigger(), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "In the constructor of a free_form topology from a graph object, an invalid edge weight of "
                           + std::to_string(-1.) + " was detected (the weight must be in the [0., 1.] range)");
    });

    bogus[res.first] = std::numeric_limits<double>::quiet_NaN();

    BOOST_CHECK_EXCEPTION(trigger(), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "In the constructor of a free_form topology from a graph object, an invalid edge weight of "
                           + std::to_string(std::numeric_limits<double>::quiet_NaN())
                           + " was detected (the weight must be in the [0., 1.] range)");
    });
}

BOOST_AUTO_TEST_CASE(udt_ctor)
{
    ring r0{100, .25};
    free_form f0{r0};

    BOOST_CHECK(f0.num_vertices() == 100u);
    BOOST_CHECK(f0.are_adjacent(0, 1));
    BOOST_CHECK(f0.are_adjacent(1, 0));
    BOOST_CHECK(f0.are_adjacent(0, 99));
    BOOST_CHECK(f0.are_adjacent(99, 0));
    BOOST_CHECK(!f0.are_adjacent(0, 2));
    BOOST_CHECK(!f0.are_adjacent(2, 0));

    for (auto i = 0u; i < 100u; ++i) {
        auto c = f0.get_connections(i);

        BOOST_CHECK(c.first.size() == 2u);
        BOOST_CHECK(c.second.size() == 2u);
        BOOST_CHECK(std::all_of(c.second.begin(), c.second.end(), [](double w) { return w == .25; }));
    }
}

BOOST_AUTO_TEST_CASE(topology_ctor)
{
    ring r0{100, .25};
    free_form f0{topology{r0}};

    BOOST_CHECK(f0.num_vertices() == 100u);
    BOOST_CHECK(f0.are_adjacent(0, 1));
    BOOST_CHECK(f0.are_adjacent(1, 0));
    BOOST_CHECK(f0.are_adjacent(0, 99));
    BOOST_CHECK(f0.are_adjacent(99, 0));
    BOOST_CHECK(!f0.are_adjacent(0, 2));
    BOOST_CHECK(!f0.are_adjacent(2, 0));

    for (auto i = 0u; i < 100u; ++i) {
        auto c = f0.get_connections(i);

        BOOST_CHECK(c.first.size() == 2u);
        BOOST_CHECK(c.second.size() == 2u);
        BOOST_CHECK(std::all_of(c.second.begin(), c.second.end(), [](double w) { return w == .25; }));
    }
}
