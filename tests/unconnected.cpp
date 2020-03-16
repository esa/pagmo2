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

#define BOOST_TEST_MODULE unconnected
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <sstream>

#include <boost/algorithm/string/predicate.hpp>

#include <pagmo/exceptions.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/topologies/ring.hpp>
#include <pagmo/topologies/unconnected.hpp>
#include <pagmo/topology.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(basic_test)
{
    unconnected r0;

    BOOST_CHECK(r0.get_connections(0).first.empty());
    BOOST_CHECK(r0.get_connections(0).second.empty());

    r0.push_back();

    BOOST_CHECK(r0.get_connections(1).first.empty());
    BOOST_CHECK(r0.get_connections(1).second.empty());

    // Minimal serialization test.
    {
        topology t0(r0);
        std::stringstream ss;
        {
            boost::archive::binary_oarchive oarchive(ss);
            oarchive << t0;
        }
        topology t1(ring{});
        BOOST_CHECK(!t1.is<unconnected>());
        {
            boost::archive::binary_iarchive iarchive(ss);
            iarchive >> t1;
        }
        BOOST_CHECK(t1.is<unconnected>());
    }
}

BOOST_AUTO_TEST_CASE(to_bgl_test)
{
    BOOST_CHECK(!has_to_bgl<unconnected>::value);

    BOOST_CHECK_EXCEPTION(
        topology{unconnected{}}.to_bgl(), not_implemented_error, [](const not_implemented_error &nie) {
            return boost::contains(
                nie.what(),
                "The to_bgl() method has been invoked, but it is not implemented in a UDT of type 'Unconnected'");
        });
}
