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

#include <pagmo/topologies/ring.hpp>

#define BOOST_TEST_MODULE ring_test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <vector>

#include <pagmo/types.hpp>

using vidx_t = std::vector<std::size_t>;

static inline vidx_t sorted(vidx_t v)
{
    std::sort(v.begin(), v.end());
    return v;
}

#include <pagmo/topology.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(ring_test)
{
    BOOST_CHECK(is_udt<ring>::value);
    ring r;
    BOOST_CHECK_EQUAL(r.num_vertices(), 0u);
    r.push_back();
    BOOST_CHECK_EQUAL(r.num_vertices(), 1u);
    auto c = r.get_connections(0u);
    BOOST_CHECK(c.first.empty());
    BOOST_CHECK(c.second.empty());
    r.push_back();
    BOOST_CHECK_EQUAL(r.num_vertices(), 2u);
    c = r.get_connections(0u);
    BOOST_CHECK(c.first.size() == 1u);
    BOOST_CHECK(c.first[0] == 1u);
    BOOST_CHECK(c.second.size() == 1u);
    BOOST_CHECK(c.second[0] == 1.);
    c = r.get_connections(1u);
    BOOST_CHECK(c.first.size() == 1u);
    BOOST_CHECK(c.first[0] == 0u);
    BOOST_CHECK(c.second.size() == 1u);
    BOOST_CHECK(c.second[0] == 1.);
    r.push_back();
    BOOST_CHECK_EQUAL(r.num_vertices(), 3u);
    c = r.get_connections(0u);
    BOOST_CHECK((sorted(c.first) == vidx_t{1u, 2u}));
    BOOST_CHECK(c.second.size() == 2u);
    BOOST_CHECK(c.second[0] == 1.);
    BOOST_CHECK(c.second[1] == 1.);
    c = r.get_connections(1u);
    BOOST_CHECK((sorted(c.first) == vidx_t{0u, 2u}));
    BOOST_CHECK(c.second.size() == 2u);
    BOOST_CHECK(c.second[0] == 1.);
    BOOST_CHECK(c.second[1] == 1.);
    c = r.get_connections(2u);
    BOOST_CHECK((sorted(c.first) == vidx_t{0u, 1u}));
    BOOST_CHECK(c.second.size() == 2u);
    BOOST_CHECK(c.second[0] == 1.);
    BOOST_CHECK(c.second[1] == 1.);
    r.push_back();
    BOOST_CHECK_EQUAL(r.num_vertices(), 4u);
    c = r.get_connections(0u);
    BOOST_CHECK((sorted(c.first) == vidx_t{1u, 3u}));
    BOOST_CHECK(c.second.size() == 2u);
    BOOST_CHECK(c.second[0] == 1.);
    BOOST_CHECK(c.second[1] == 1.);
    c = r.get_connections(1u);
    BOOST_CHECK((sorted(c.first) == vidx_t{0u, 2u}));
    BOOST_CHECK(c.second.size() == 2u);
    BOOST_CHECK(c.second[0] == 1.);
    BOOST_CHECK(c.second[1] == 1.);
    c = r.get_connections(2u);
    BOOST_CHECK((sorted(c.first) == vidx_t{1u, 3u}));
    BOOST_CHECK(c.second.size() == 2u);
    BOOST_CHECK(c.second[0] == 1.);
    BOOST_CHECK(c.second[1] == 1.);
    c = r.get_connections(3u);
    BOOST_CHECK((sorted(c.first) == vidx_t{0u, 2u}));
    BOOST_CHECK(c.second.size() == 2u);
    BOOST_CHECK(c.second[0] == 1.);
    BOOST_CHECK(c.second[1] == 1.);
    r.push_back();
    BOOST_CHECK_EQUAL(r.num_vertices(), 5u);
    c = r.get_connections(0u);
    BOOST_CHECK((sorted(c.first) == vidx_t{1u, 4u}));
    BOOST_CHECK((c.second == vector_double{1., 1.}));
    c = r.get_connections(1u);
    BOOST_CHECK((sorted(c.first) == vidx_t{0u, 2u}));
    BOOST_CHECK((c.second == vector_double{1., 1.}));
    c = r.get_connections(2u);
    BOOST_CHECK((sorted(c.first) == vidx_t{1u, 3u}));
    BOOST_CHECK((c.second == vector_double{1., 1.}));
    c = r.get_connections(3u);
    BOOST_CHECK((sorted(c.first) == vidx_t{2u, 4u}));
    BOOST_CHECK((c.second == vector_double{1., 1.}));
    c = r.get_connections(4u);
    BOOST_CHECK((sorted(c.first) == vidx_t{0u, 3u}));
    BOOST_CHECK((c.second == vector_double{1., 1.}));
}
