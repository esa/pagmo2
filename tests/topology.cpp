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

#define BOOST_TEST_MODULE topology_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <pagmo/s11n.hpp>
#include <pagmo/topologies/unconnected.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

struct gc00 {
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const;
};

struct ngc00 {
    void get_connections(std::size_t) const;
};

struct ngc01 {
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t);
};

struct pb00 {
    void push_back();
};

struct npb00 {
};

struct npb01 {
    int push_back();
};

struct udt00 {
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const
    {
        return {{0, 1, 2}, {0.1, 0.2, 0.3}};
    }
    void push_back()
    {
        ++n_pushed;
    }
    std::string get_name() const
    {
        return "udt00";
    }
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &n_pushed;
    }
    int n_pushed = 0;
};

PAGMO_S11N_TOPOLOGY_EXPORT(udt00)

struct udt01 {
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const
    {
        return {{3, 4, 5}, {0.1, 0.2}};
    }
    void push_back()
    {
        ++n_pushed;
    }
    std::string get_extra_info() const
    {
        return "hello";
    }
    int n_pushed = 0;
};

BOOST_AUTO_TEST_CASE(topology_type_traits_test)
{
    BOOST_CHECK(!has_get_connections<void>::value);
    BOOST_CHECK(has_get_connections<gc00>::value);
    BOOST_CHECK(!has_get_connections<ngc00>::value);
    BOOST_CHECK(!has_get_connections<ngc01>::value);

    BOOST_CHECK(!has_push_back<void>::value);
    BOOST_CHECK(has_push_back<pb00>::value);
    BOOST_CHECK(!has_push_back<npb00>::value);
    BOOST_CHECK(!has_push_back<npb01>::value);

    BOOST_CHECK(!is_udt<void>::value);
    BOOST_CHECK(is_udt<udt00>::value);
    BOOST_CHECK(!is_udt<gc00>::value);
    BOOST_CHECK(!is_udt<pb00>::value);
}

BOOST_AUTO_TEST_CASE(topology_basic_tests)
{
    topology def00;
    BOOST_CHECK(def00.is<unconnected>());

    BOOST_CHECK((!std::is_constructible<topology, int>::value));
    BOOST_CHECK((!std::is_constructible<topology, pb00>::value));

    topology t0{udt00{}}, t1{udt01{}};

    BOOST_CHECK(t0.is_valid());
    BOOST_CHECK(t0.is<udt00>());
    BOOST_CHECK(!t0.is<udt01>());
    BOOST_CHECK(t0.extract<udt00>() != nullptr);
    BOOST_CHECK(static_cast<const topology &>(t0).extract<udt00>() != nullptr);
    BOOST_CHECK(t0.extract<udt01>() == nullptr);
    BOOST_CHECK(static_cast<const topology &>(t0).extract<udt01>() == nullptr);
    BOOST_CHECK(t0.get_name() == "udt00");
    BOOST_CHECK(t0.get_extra_info().empty());

    t0.push_back();
    t0.push_back();

    BOOST_CHECK(t0.extract<udt00>()->n_pushed == 2);

    // Copy construction.
    auto t3(t0);
    BOOST_CHECK(t3.is_valid());
    BOOST_CHECK(t3.is<udt00>());
    BOOST_CHECK(t3.extract<udt00>()->n_pushed == 2);
    BOOST_CHECK(static_cast<const topology &>(t3).extract<udt00>()->n_pushed == 2);
    BOOST_CHECK(t3.get_name() == "udt00");
    BOOST_CHECK(t3.get_extra_info().empty());

    // Copy assignment.
    topology t4;
    t4 = t3;
    BOOST_CHECK(t4.is_valid());
    BOOST_CHECK(t4.is<udt00>());
    BOOST_CHECK(t4.extract<udt00>()->n_pushed == 2);
    BOOST_CHECK(static_cast<const topology &>(t4).extract<udt00>()->n_pushed == 2);
    BOOST_CHECK(t4.get_name() == "udt00");

    // Move construction.
    auto t5(std::move(t4));
    BOOST_CHECK(!t4.is_valid());
    BOOST_CHECK(t5.is_valid());
    BOOST_CHECK(t5.is<udt00>());
    BOOST_CHECK(t5.extract<udt00>()->n_pushed == 2);
    BOOST_CHECK(t5.get_name() == "udt00");
    BOOST_CHECK(t5.get_extra_info().empty());

    // Move assignment.
    topology t6;
    t6 = std::move(t5);
    BOOST_CHECK(!t5.is_valid());
    BOOST_CHECK(t6.is_valid());
    BOOST_CHECK(t6.is<udt00>());
    BOOST_CHECK(t6.extract<udt00>()->n_pushed == 2);
    BOOST_CHECK(t6.get_name() == "udt00");

    // Generic assignment.
    BOOST_CHECK((!std::is_assignable<topology &, int>::value));
    t6 = udt01{};
    BOOST_CHECK(t6.is_valid());
    BOOST_CHECK(t6.is<udt01>());
    BOOST_CHECK(t6.extract<udt01>()->n_pushed == 0);
    BOOST_CHECK(t6.get_extra_info() == "hello");
}

// Bad connections.
struct bc00 : udt00 {
    // Inconsistent vector sizes.
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const
    {
        return {{0, 1}, {0.1, 0.2, 0.3}};
    }
};

struct bc01 : udt00 {
    // Non-finite weight.
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const
    {
        return {{0, 1}, {0.1, std::numeric_limits<double>::infinity()}};
    }
};

struct bc02 : udt00 {
    // Weight outside the probability range.
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const
    {
        return {{0, 1}, {0.1, 2.}};
    }
};

BOOST_AUTO_TEST_CASE(topology_get_connections_test)
{
    topology t0{udt00{}};
    BOOST_CHECK(t0.get_connections(0) == t0.get_connections(1));
    BOOST_CHECK((t0.get_connections(0).first == std::vector<std::size_t>{0, 1, 2}));
    BOOST_CHECK((t0.get_connections(0).second == std::vector<double>{.1, .2, .3}));

    t0 = bc00{};

    BOOST_CHECK_EXCEPTION(t0.get_connections(0), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(),
                               "An invalid pair of vectors was returned by the 'get_connections()' method "
                               "of the 'udt00' topology: the vector of connecting islands has a size of 2, while the "
                               "vector of migration probabilities has a size of 3 (the two sizes must be equal)");
    });

    t0 = bc01{};

    BOOST_CHECK_EXCEPTION(t0.get_connections(0), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(),
            "An invalid non-finite migration probability of " + std::to_string(std::numeric_limits<double>::infinity())
                + " was detected in the vector of migration probabilities returned by the 'get_connections()' "
                  "method of the 'udt00' topology");
    });

    t0 = bc02{};

    BOOST_CHECK_EXCEPTION(t0.get_connections(0), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(),
            "An invalid migration probability of " + std::to_string(2.)
                + " was detected in the vector of migration probabilities returned by the 'get_connections()' "
                  "method of the 'udt00' topology: the value must be in the [0., 1.] range");
    });
}

BOOST_AUTO_TEST_CASE(topology_s11n_test)
{
    topology t0{udt00{}};
    t0.push_back();
    t0.push_back();

    std::stringstream ss;
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << t0;
    }
    topology t1;
    BOOST_CHECK(!t1.is<udt00>());
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> t1;
    }

    BOOST_CHECK(t1.is<udt00>());
    BOOST_CHECK(t1.get_name() == "udt00");
    BOOST_CHECK(t1.extract<udt00>()->n_pushed == 2);
}

BOOST_AUTO_TEST_CASE(topology_stream_test)
{
    {
        topology t0{udt01{}};

        std::ostringstream oss;

        oss << t0;

        auto str = oss.str();

        BOOST_CHECK(boost::contains(str, "Topology name:"));
        BOOST_CHECK(boost::contains(str, "hello"));
    }

    {
        topology t0{udt00{}};

        std::ostringstream oss;

        oss << t0;

        auto str = oss.str();

        BOOST_CHECK(boost::contains(str, "Topology name: udt00"));
    }
}
