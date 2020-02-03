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

#if defined(_MSC_VER)

#define _SCL_SECURE_NO_WARNINGS

#endif

#define BOOST_TEST_MODULE archipelago_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

#include <pagmo/algorithms/de.hpp>
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/island.hpp>
#include <pagmo/islands/thread_island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/r_policies/fair_replace.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/s_policies/select_best.hpp>
#include <pagmo/topologies/fully_connected.hpp>
#include <pagmo/topologies/ring.hpp>
#include <pagmo/topologies/unconnected.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(archipelago_construction)
{
    // Make the test deterministic.
    random_device::set_seed(123u);

    using size_type = archipelago::size_type;
    archipelago archi;

    std::cout << archi << '\n';

    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);
    BOOST_CHECK(archi.get_topology().is<unconnected>());
    BOOST_CHECK(archi.size() == 0u);
    BOOST_CHECK(archi.get_migration_log().empty());
    BOOST_CHECK(archi.get_migrants_db().empty());

    // Copy constructor.
    archi.set_topology(topology{ring{}});
    archi.set_migration_type(migration_type::broadcast);
    archi.set_migrant_handling(migrant_handling::evict);
    auto archi_copy(archi);
    BOOST_CHECK(archi_copy.get_migration_type() == migration_type::broadcast);
    BOOST_CHECK(archi_copy.get_migrant_handling() == migrant_handling::evict);
    BOOST_CHECK(archi_copy.get_topology().is<ring>());
    BOOST_CHECK(archi_copy.size() == 0u);
    BOOST_CHECK(archi_copy.get_migration_log().empty());
    BOOST_CHECK(archi_copy.get_migrants_db().empty());

    // Move constructor.
    auto archi_move(std::move(archi_copy));
    BOOST_CHECK(archi_move.get_migration_type() == migration_type::broadcast);
    BOOST_CHECK(archi_move.get_migrant_handling() == migrant_handling::evict);
    BOOST_CHECK(archi_move.get_topology().is<ring>());
    BOOST_CHECK(archi_move.size() == 0u);
    BOOST_CHECK(archi_move.get_migration_log().empty());
    BOOST_CHECK(archi_move.get_migrants_db().empty());

    archipelago archi2(0u, de{}, rosenbrock{}, 10u);
    BOOST_CHECK(archi2.size() == 0u);
    BOOST_CHECK(archi2.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi2.get_migrant_handling() == migrant_handling::preserve);
    archipelago archi3(5u, de{}, rosenbrock{}, 10u);
    BOOST_CHECK(archi3.size() == 5u);
    BOOST_CHECK(archi3.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi3.get_migrant_handling() == migrant_handling::preserve);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi3[i].status() != evolve_status::busy);
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
    }
    archi3 = archipelago{5u, thread_island{}, de{}, rosenbrock{}, 10u};
    BOOST_CHECK(archi3.size() == 5u);
    BOOST_CHECK(archi3.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi3.get_migrant_handling() == migrant_handling::preserve);
    std::vector<unsigned> seeds;
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi3[i].status() != evolve_status::busy);
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
        BOOST_CHECK(archi3[i].get_r_policy().is<fair_replace>());
        BOOST_CHECK(archi3[i].get_s_policy().is<select_best>());
        seeds.push_back(archi3[i].get_population().get_seed());
    }
    // Check seeds are different (not guaranteed but very likely).
    std::sort(seeds.begin(), seeds.end());
    BOOST_CHECK(std::unique(seeds.begin(), seeds.end()) == seeds.end());
    archi3 = archipelago{5u, thread_island{}, de{}, population{rosenbrock{}, 10u}};
    BOOST_CHECK(archi3.size() == 5u);
    BOOST_CHECK(archi3.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi3.get_migrant_handling() == migrant_handling::preserve);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi3[i].status() != evolve_status::busy);
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
    }
    archi3 = archipelago{5u, thread_island{}, de{}, population{rosenbrock{}, 10u, 123u}};
    BOOST_CHECK(archi3.size() == 5u);
    BOOST_CHECK(archi3.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi3.get_migrant_handling() == migrant_handling::preserve);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi3[i].status() != evolve_status::busy);
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
    }
    // A couple of tests for the constructor which contains a seed argument.
    archipelago archi3a{5u, thread_island{}, de{}, rosenbrock{}, 10u, 123u};
    BOOST_CHECK(archi3a.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi3a.get_migrant_handling() == migrant_handling::preserve);
    seeds.clear();
    std::transform(archi3a.begin(), archi3a.end(), std::back_inserter(seeds),
                   [](const island &isl) { return isl.get_population().get_seed(); });
    std::sort(seeds.begin(), seeds.end());
    BOOST_CHECK(std::unique(seeds.begin(), seeds.end()) == seeds.end());
    std::vector<unsigned> seeds2;
    archipelago archi3b{5u, de{}, rosenbrock{}, 10u, 123u};
    BOOST_CHECK(archi3b.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi3b.get_migrant_handling() == migrant_handling::preserve);
    std::transform(archi3b.begin(), archi3b.end(), std::back_inserter(seeds2),
                   [](const island &isl) { return isl.get_population().get_seed(); });
    std::sort(seeds2.begin(), seeds2.end());
    BOOST_CHECK(std::unique(seeds2.begin(), seeds2.end()) == seeds2.end());
    BOOST_CHECK(std::equal(seeds.begin(), seeds.end(), seeds2.begin()));
    BOOST_CHECK(
        std::equal(archi3a.begin(), archi3a.end(), archi3b.begin(), [](const island &isl_a, const island &isl_b) {
            return isl_a.get_population().get_x() == isl_b.get_population().get_x();
        }));
    BOOST_CHECK(
        std::equal(archi3a.begin(), archi3a.end(), archi3b.begin(), [](const island &isl_a, const island &isl_b) {
            return isl_a.get_population().get_f() == isl_b.get_population().get_f();
        }));
    BOOST_CHECK(
        std::equal(archi3a.begin(), archi3a.end(), archi3b.begin(), [](const island &isl_a, const island &isl_b) {
            return isl_a.get_population().get_ID() == isl_b.get_population().get_ID();
        }));
    archi3a = archipelago{5u, thread_island{}, de{}, rosenbrock{}, 10u};
    archi3b = archipelago{5u, thread_island{}, de{}, rosenbrock{}, 10u};
    BOOST_CHECK(
        !std::equal(archi3a.begin(), archi3a.end(), archi3b.begin(), [](const island &isl_a, const island &isl_b) {
            return isl_a.get_population().get_x() == isl_b.get_population().get_x();
        }));
    BOOST_CHECK(
        !std::equal(archi3a.begin(), archi3a.end(), archi3b.begin(), [](const island &isl_a, const island &isl_b) {
            return isl_a.get_population().get_f() == isl_b.get_population().get_f();
        }));
    BOOST_CHECK(
        !std::equal(archi3a.begin(), archi3a.end(), archi3b.begin(), [](const island &isl_a, const island &isl_b) {
            return isl_a.get_population().get_ID() == isl_b.get_population().get_ID();
        }));
    auto archi4 = archi3;
    BOOST_CHECK(archi4.size() == 5u);
    BOOST_CHECK(archi4.get_migrants_db().size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi4[i].status() != evolve_status::busy);
        BOOST_CHECK(archi4[i].get_algorithm().is<de>());
        BOOST_CHECK(archi4[i].get_population().size() == 10u);
        BOOST_CHECK(archi4[i].get_population().get_problem().is<rosenbrock>());
    }
    archi4.evolve(10);
    auto archi5 = archi4;
    BOOST_CHECK(archi5.size() == 5u);
    BOOST_CHECK(archi5.get_migrants_db().size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi5[i].status() != evolve_status::busy);
        BOOST_CHECK(archi5[i].get_algorithm().is<de>());
        BOOST_CHECK(archi5[i].get_population().size() == 10u);
        BOOST_CHECK(archi5[i].get_population().get_problem().is<rosenbrock>());
    }
    archi4.wait_check();
    BOOST_CHECK(archi4.status() == evolve_status::idle);
    archi4.evolve(10);
    auto archi6(std::move(archi4));
    BOOST_CHECK(archi6.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi6[i].status() != evolve_status::busy);
        BOOST_CHECK(archi6[i].get_algorithm().is<de>());
        BOOST_CHECK(archi6[i].get_population().size() == 10u);
        BOOST_CHECK(archi6[i].get_population().get_problem().is<rosenbrock>());
    }
    BOOST_CHECK(archi4.size() == 0u);
    archi4 = archi5;
    BOOST_CHECK(archi4.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi4[i].status() != evolve_status::busy);
        BOOST_CHECK(archi4[i].get_algorithm().is<de>());
        BOOST_CHECK(archi4[i].get_population().size() == 10u);
        BOOST_CHECK(archi4[i].get_population().get_problem().is<rosenbrock>());
    }
    archi4 = std::move(archi5);
    BOOST_CHECK(archi4.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi4[i].status() != evolve_status::busy);
        BOOST_CHECK(archi4[i].get_algorithm().is<de>());
        BOOST_CHECK(archi4[i].get_population().size() == 10u);
        BOOST_CHECK(archi4[i].get_population().get_problem().is<rosenbrock>());
    }
    BOOST_CHECK(archi5.size() == 0u);
    // Self assignment.
    archi4 = *&archi4;
    BOOST_CHECK((std::is_same<archipelago &, decltype(archi4 = archi4)>::value));
    BOOST_CHECK(archi4.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi4[i].status() != evolve_status::busy);
        BOOST_CHECK(archi4[i].get_algorithm().is<de>());
        BOOST_CHECK(archi4[i].get_population().size() == 10u);
        BOOST_CHECK(archi4[i].get_population().get_problem().is<rosenbrock>());
    }
#if !defined(__clang__)
    archi4 = std::move(archi4);
    BOOST_CHECK((std::is_same<archipelago &, decltype(archi4 = std::move(archi4))>::value));
    BOOST_CHECK(archi4.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi4[i].status() != evolve_status::busy);
        BOOST_CHECK(archi4[i].get_algorithm().is<de>());
        BOOST_CHECK(archi4[i].get_population().size() == 10u);
        BOOST_CHECK(archi4[i].get_population().get_problem().is<rosenbrock>());
    }
#endif
}

struct udrp00 {
    individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double &, const individuals_group_t &) const
    {
        return individuals_group_t{};
    }
};

struct udsp00 {
    individuals_group_t select(const individuals_group_t &, const vector_double::size_type &,
                               const vector_double::size_type &, const vector_double::size_type &,
                               const vector_double::size_type &, const vector_double::size_type &,
                               const vector_double &) const
    {
        return individuals_group_t{};
    }
};

BOOST_AUTO_TEST_CASE(archipelago_policy_constructors)
{
    // algo, prob, size, policies, no seed.
    archipelago archi{5u, de{}, rosenbrock{}, 10u, udrp00{}, udsp00{}};
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);
    BOOST_CHECK(archi.size() == 5u);
    for (const auto &isl : archi) {
        BOOST_CHECK(isl.get_r_policy().is<udrp00>());
        BOOST_CHECK(isl.get_s_policy().is<udsp00>());
    }

    // algo, prob, size, policies, seed.
    std::vector<unsigned> seeds;
    archi = archipelago{5u, de{}, rosenbrock{}, 10u, udrp00{}, udsp00{}, 5};
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);
    BOOST_CHECK(archi.size() == 5u);
    BOOST_CHECK(archi.get_migrants_db().size() == 5u);
    for (const auto &isl : archi) {
        BOOST_CHECK(isl.get_r_policy().is<udrp00>());
        BOOST_CHECK(isl.get_s_policy().is<udsp00>());
        seeds.push_back(isl.get_population().get_seed());
    }
    // Check seeds are different (not guaranteed but very likely).
    std::sort(seeds.begin(), seeds.end());
    BOOST_CHECK(std::unique(seeds.begin(), seeds.end()) == seeds.end());

    // algo, prob, bfe, rpol, spol, no seed.
    archi = archipelago{5u, de{}, rosenbrock{}, bfe{}, 10u, udrp00{}, udsp00{}};
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);
    BOOST_CHECK(archi.size() == 5u);
    for (const auto &isl : archi) {
        BOOST_CHECK(isl.get_r_policy().is<udrp00>());
        BOOST_CHECK(isl.get_s_policy().is<udsp00>());
    }

    // algo, prob, bfe, rpol, spol, seed.
    seeds.clear();
    archi = archipelago{5u, de{}, rosenbrock{}, bfe{}, 10u, udrp00{}, udsp00{}, 5};
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);
    BOOST_CHECK(archi.size() == 5u);
    for (const auto &isl : archi) {
        BOOST_CHECK(isl.get_r_policy().is<udrp00>());
        BOOST_CHECK(isl.get_s_policy().is<udsp00>());
        seeds.push_back(isl.get_population().get_seed());
    }
    std::sort(seeds.begin(), seeds.end());
    BOOST_CHECK(std::unique(seeds.begin(), seeds.end()) == seeds.end());

    // isl, algo, prob, rpol, spol, no seed.
    archi = archipelago{5u, thread_island{}, de{}, rosenbrock{}, 10u, udrp00{}, udsp00{}};
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);
    BOOST_CHECK(archi.size() == 5u);
    for (const auto &isl : archi) {
        BOOST_CHECK(isl.get_r_policy().is<udrp00>());
        BOOST_CHECK(isl.get_s_policy().is<udsp00>());
    }

    // isl, algo, prob, rpol, spol, seed.
    seeds.clear();
    archi = archipelago{5u, thread_island{}, de{}, rosenbrock{}, 10u, udrp00{}, udsp00{}, 5};
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);
    BOOST_CHECK(archi.size() == 5u);
    BOOST_CHECK(archi.get_migrants_db().size() == 5u);
    for (const auto &isl : archi) {
        BOOST_CHECK(isl.get_r_policy().is<udrp00>());
        BOOST_CHECK(isl.get_s_policy().is<udsp00>());
        seeds.push_back(isl.get_population().get_seed());
    }
    std::sort(seeds.begin(), seeds.end());
    BOOST_CHECK(std::unique(seeds.begin(), seeds.end()) == seeds.end());

    // isl, algo, prob, bfe, rpol, spol, no seed.
    archi = archipelago{5u, thread_island{}, de{}, rosenbrock{}, bfe{}, 10u, udrp00{}, udsp00{}};
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);
    BOOST_CHECK(archi.size() == 5u);
    for (const auto &isl : archi) {
        BOOST_CHECK(isl.get_r_policy().is<udrp00>());
        BOOST_CHECK(isl.get_s_policy().is<udsp00>());
    }

    // isl, algo, prob, bfe, rpol, spol, seed.
    seeds.clear();
    archi = archipelago{5u, thread_island{}, de{}, rosenbrock{}, bfe{}, 10u, udrp00{}, udsp00{}, 5};
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);
    BOOST_CHECK(archi.size() == 5u);
    BOOST_CHECK(archi.get_migrants_db().size() == 5u);
    for (const auto &isl : archi) {
        BOOST_CHECK(isl.get_r_policy().is<udrp00>());
        BOOST_CHECK(isl.get_s_policy().is<udsp00>());
        seeds.push_back(isl.get_population().get_seed());
    }
    std::sort(seeds.begin(), seeds.end());
    BOOST_CHECK(std::unique(seeds.begin(), seeds.end()) == seeds.end());
}

BOOST_AUTO_TEST_CASE(archipelago_topology_constructors)
{
    archipelago archi{topology{}};

    BOOST_CHECK(archi.get_topology().is<unconnected>());
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);

    archi = archipelago{ring{}};
    BOOST_CHECK(archi.get_topology().is<ring>());
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);

    // Check that the topology is preserved in copy/move ops.
    auto archi2(archi);
    BOOST_CHECK(archi2.get_topology().is<ring>());
    BOOST_CHECK(archi2.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi2.get_migrant_handling() == migrant_handling::preserve);

    auto archi3(std::move(archi2));
    BOOST_CHECK(archi3.get_topology().is<ring>());
    BOOST_CHECK(archi3.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi3.get_migrant_handling() == migrant_handling::preserve);

    archipelago archi4;
    archi4 = archi3;
    BOOST_CHECK(archi4.get_topology().is<ring>());
    BOOST_CHECK(archi4.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi4.get_migrant_handling() == migrant_handling::preserve);

    archipelago archi5;
    archi5 = std::move(archi4);
    BOOST_CHECK(archi5.get_topology().is<ring>());
    BOOST_CHECK(archi5.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi5.get_migrant_handling() == migrant_handling::preserve);

    // Ctors from topology and number of islands.
    archi = archipelago{topology{}, 5};
    BOOST_CHECK(archi.size() == 5u);
    BOOST_CHECK(archi.get_topology().is<unconnected>());
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);

    // Invoke one of the complicated ctors
    archi = archipelago{ring{}, 5u, thread_island{}, de{}, rosenbrock{}, bfe{}, 10u, udrp00{}, udsp00{}, 5};
    BOOST_CHECK(archi.size() == 5u);
    BOOST_CHECK(archi.get_topology().is<ring>());
    BOOST_CHECK(archi.get_migration_type() == migration_type::p2p);
    BOOST_CHECK(archi.get_migrant_handling() == migrant_handling::preserve);
    std::vector<unsigned> seeds;
    for (const auto &isl : archi) {
        BOOST_CHECK(isl.is<thread_island>());
        BOOST_CHECK(isl.get_algorithm().is<de>());
        BOOST_CHECK(isl.get_population().get_problem().is<rosenbrock>());
        BOOST_CHECK(isl.get_population().size() == 10u);
        BOOST_CHECK(isl.get_r_policy().is<udrp00>());
        BOOST_CHECK(isl.get_s_policy().is<udsp00>());
        seeds.push_back(isl.get_population().get_seed());
    }
    std::sort(seeds.begin(), seeds.end());
    BOOST_CHECK(std::unique(seeds.begin(), seeds.end()) == seeds.end());
}

BOOST_AUTO_TEST_CASE(archipelago_push_back_migr)
{
    // Verify that push_back() also calls the topology's push_back() and adds
    // entries to the migrants db.
    archipelago archi{ring{}};

    BOOST_CHECK(archi.get_topology().extract<ring>()->num_vertices() == 0u);
    BOOST_CHECK(archi.get_migrants_db().size() == 0u);

    archi.push_back();

    BOOST_CHECK(archi.get_topology().extract<ring>()->num_vertices() == 1u);
    BOOST_CHECK(!archi.get_topology().extract<ring>()->are_adjacent(0, 0));
    BOOST_CHECK(archi.get_migrants_db().size() == 1u);

    archi.push_back();
    BOOST_CHECK(archi.get_topology().extract<ring>()->num_vertices() == 2u);
    BOOST_CHECK(archi.get_topology().extract<ring>()->are_adjacent(1, 0));
    BOOST_CHECK(archi.get_topology().extract<ring>()->are_adjacent(0, 1));
    BOOST_CHECK(archi.get_migrants_db().size() == 2u);

    archi.push_back();
    BOOST_CHECK(archi.get_topology().extract<ring>()->num_vertices() == 3u);
    BOOST_CHECK(archi.get_topology().extract<ring>()->are_adjacent(0, 1));
    BOOST_CHECK(archi.get_topology().extract<ring>()->are_adjacent(1, 0));
    BOOST_CHECK(archi.get_topology().extract<ring>()->are_adjacent(1, 2));
    BOOST_CHECK(archi.get_topology().extract<ring>()->are_adjacent(2, 1));
    BOOST_CHECK(archi.get_topology().extract<ring>()->are_adjacent(0, 2));
    BOOST_CHECK(archi.get_topology().extract<ring>()->are_adjacent(2, 0));
    BOOST_CHECK(archi.get_migrants_db().size() == 3u);
}

BOOST_AUTO_TEST_CASE(archipelago_topology_setter)
{
    archipelago archi{ring{}};

    archi.push_back();
    archi.push_back();
    archi.push_back();
    archi.push_back();

    BOOST_CHECK(archi.get_topology().is<ring>());

    topology new_top{fully_connected{}};
    new_top.push_back();
    new_top.push_back();
    new_top.push_back();
    new_top.push_back();

    archi.set_topology(new_top);

    BOOST_CHECK(archi.get_topology().is<fully_connected>());

    BOOST_CHECK((archi.get_topology().get_connections(0).first == std::vector<std::size_t>{1, 2, 3}));
    BOOST_CHECK((archi.get_topology().get_connections(1).first == std::vector<std::size_t>{0, 2, 3}));
    BOOST_CHECK((archi.get_topology().get_connections(2).first == std::vector<std::size_t>{0, 1, 3}));
    BOOST_CHECK((archi.get_topology().get_connections(3).first == std::vector<std::size_t>{0, 1, 2}));
}

BOOST_AUTO_TEST_CASE(archipelago_island_access)
{
    archipelago archi0;
    BOOST_CHECK_THROW(archi0[0], std::out_of_range);
    BOOST_CHECK_THROW(static_cast<archipelago const &>(archi0)[0], std::out_of_range);
    archi0.push_back(de{}, rosenbrock{}, 10u);
    archi0.push_back(pso{}, schwefel{4}, 11u);
    BOOST_CHECK(archi0[0].get_algorithm().is<de>());
    BOOST_CHECK(static_cast<archipelago const &>(archi0)[1].get_algorithm().is<pso>());
    BOOST_CHECK(archi0[0].get_population().size() == 10u);
    BOOST_CHECK(archi0[1].get_population().size() == 11u);
    BOOST_CHECK(static_cast<archipelago const &>(archi0)[0].get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(archi0[1].get_population().get_problem().is<schwefel>());
    island &i0 = archi0[0];
    const island &i1 = archi0[1];
    archi0.push_back(thread_island{}, de{}, schwefel{12}, 12u);
    BOOST_CHECK(i0.get_algorithm().is<de>());
    BOOST_CHECK(i1.get_algorithm().is<pso>());
    BOOST_CHECK(i0.get_population().size() == 10u);
    BOOST_CHECK(i1.get_population().size() == 11u);
    BOOST_CHECK(i0.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(i1.get_population().get_problem().is<schwefel>());
    BOOST_CHECK(archi0[2].get_algorithm().is<de>());
    BOOST_CHECK(archi0[2].get_population().get_problem().is<schwefel>());
    BOOST_CHECK_THROW(archi0[3], std::out_of_range);
    BOOST_CHECK_THROW(static_cast<archipelago const &>(archi0)[3], std::out_of_range);
}

BOOST_AUTO_TEST_CASE(archipelago_evolve)
{
    archipelago archi(10u, de{}, rosenbrock{20}, 20u), archi3;
    archi.evolve(10);
    {
        // Copy while evolving.
        auto archi2(archi);
        archi3 = archi;
        archi.wait_check();
        BOOST_CHECK(archi.status() == evolve_status::idle);
        BOOST_CHECK(archi3.status() != evolve_status::busy);
        BOOST_CHECK(archi2.status() != evolve_status::busy);
        BOOST_CHECK(archi2.size() == 10u);
        BOOST_CHECK(archi3.size() == 10u);
        BOOST_CHECK(archi2[2].get_algorithm().is<de>());
        BOOST_CHECK(archi3[2].get_algorithm().is<de>());
        BOOST_CHECK(archi2[2].get_population().size() == 20u);
        BOOST_CHECK(archi3[2].get_population().size() == 20u);
        BOOST_CHECK(archi2[2].get_population().get_problem().is<rosenbrock>());
        BOOST_CHECK(archi3[2].get_population().get_problem().is<rosenbrock>());
    }
    auto archi_b(archi);
    archi.evolve(10);
    archi_b.evolve(10);
    {
        // Move while evolving.
        auto archi2(std::move(archi));
        archi3 = std::move(archi_b);
        archi.wait_check();
        BOOST_CHECK(archi.status() == evolve_status::idle);
        BOOST_CHECK(archi2.size() == 10u);
        BOOST_CHECK(archi3.size() == 10u);
        BOOST_CHECK(archi2[2].get_algorithm().is<de>());
        BOOST_CHECK(archi3[2].get_algorithm().is<de>());
        BOOST_CHECK(archi2[2].get_population().size() == 20u);
        BOOST_CHECK(archi3[2].get_population().size() == 20u);
        BOOST_CHECK(archi2[2].get_population().get_problem().is<rosenbrock>());
        BOOST_CHECK(archi3[2].get_population().get_problem().is<rosenbrock>());
    }
}

static std::atomic_bool flag = ATOMIC_VAR_INIT(false);

struct prob_01 {
    vector_double fitness(const vector_double &) const
    {
        while (!flag.load()) {
        }
        return {.5};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
};

BOOST_AUTO_TEST_CASE(archipelago_get_wait_busy)
{
    flag.store(true);
    archipelago a{10, de{}, population{prob_01{}, 25}};
    BOOST_CHECK(a.status() != evolve_status::busy);
    flag.store(false);
    a.evolve();
    BOOST_CHECK(a.status() == evolve_status::busy);
    flag.store(true);
    a.wait();
    BOOST_CHECK(a.status() == evolve_status::idle);
    flag.store(false);
    a = archipelago{10, de{}, population{rosenbrock{}, 3}};
    a.evolve(10);
    a.evolve(10);
    a.evolve(10);
    a.evolve(10);
    BOOST_CHECK_THROW(a.wait_check(), std::invalid_argument);
    BOOST_CHECK(a.status() == evolve_status::idle);
    a.wait_check();
    a.wait();
}

BOOST_AUTO_TEST_CASE(archipelago_stream)
{
    archipelago a{10, de{}, population{rosenbrock{}, 25}};
    std::ostringstream oss;
    oss << a;
    BOOST_CHECK(!oss.str().empty());
    BOOST_CHECK(boost::contains(oss.str(), "Topology:"));
    BOOST_CHECK(boost::contains(oss.str(), "Migration type:"));
    BOOST_CHECK(boost::contains(oss.str(), "Migrant handling policy:"));
}

BOOST_AUTO_TEST_CASE(archipelago_serialization)
{
    archipelago a{ring{}, 10, de{}, population{rosenbrock{}, 25}};
    a.evolve(10);
    a.wait_check();
    BOOST_CHECK(a.status() == evolve_status::idle);
    a.set_migration_type(migration_type::broadcast);
    a.set_migrant_handling(migrant_handling::evict);
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(a);
    const auto mig_db_before = a.get_migrants_db();
    const auto mig_log_before = a.get_migration_log();
    // Now serialize, deserialize and compare the result.
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << a;
    }
    a = archipelago{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> a;
    }
    auto after = boost::lexical_cast<std::string>(a);
    BOOST_CHECK_EQUAL(before, after);
    BOOST_CHECK(a.get_migration_type() == migration_type::broadcast);
    BOOST_CHECK(a.get_migrant_handling() == migrant_handling::evict);
    BOOST_CHECK(a.get_migrants_db() == mig_db_before);
    BOOST_CHECK(a.get_migration_log() == mig_log_before);
    BOOST_CHECK(a.get_topology().is<ring>());
    BOOST_CHECK(a.get_topology().extract<ring>()->num_vertices() == 10u);
}

BOOST_AUTO_TEST_CASE(archipelago_iterator_tests)
{
    archipelago archi;
    BOOST_CHECK(archi.begin() == archi.end());
    BOOST_CHECK(static_cast<const archipelago &>(archi).begin() == archi.end());
    BOOST_CHECK(static_cast<const archipelago &>(archi).end() == archi.begin());
    BOOST_CHECK(static_cast<const archipelago &>(archi).begin() == static_cast<const archipelago &>(archi).end());
    BOOST_CHECK_EQUAL(std::distance(archi.begin(), archi.end()), 0);
    BOOST_CHECK_EQUAL(std::distance(std::begin(archi), std::end(archi)), 0);
    BOOST_CHECK_EQUAL(
        std::distance(static_cast<const archipelago &>(archi).begin(), static_cast<const archipelago &>(archi).end()),
        0);
    BOOST_CHECK_EQUAL(std::distance(std::begin(static_cast<const archipelago &>(archi)),
                                    std::end(static_cast<const archipelago &>(archi))),
                      0);
    archi.push_back(de{}, rosenbrock{}, 10u);
    archi.push_back(de{}, rosenbrock{}, 10u);
    archi.push_back(de{}, rosenbrock{}, 10u);
    archi.push_back(de{}, rosenbrock{}, 10u);
    BOOST_CHECK_EQUAL(std::distance(std::begin(archi), std::end(archi)), 4);
    BOOST_CHECK_EQUAL(std::distance(std::begin(static_cast<const archipelago &>(archi)),
                                    std::end(static_cast<const archipelago &>(archi))),
                      4);
    for (auto &isl : archi) {
        BOOST_CHECK_EQUAL(isl.get_population().size(), 10u);
    }
    for (const auto &isl : static_cast<const archipelago &>(archi)) {
        BOOST_CHECK_EQUAL(isl.get_population().size(), 10u);
    }
    BOOST_CHECK(archi.begin() + 4 == archi.end());
}

BOOST_AUTO_TEST_CASE(archipelago_champion_tests)
{
    archipelago archi;
    BOOST_CHECK(archi.get_champions_f().empty());
    BOOST_CHECK(archi.get_champions_x().empty());
    archi.push_back(de{}, rosenbrock{}, 20u);
    archi.push_back(de{}, rosenbrock{}, 20u);
    archi.push_back(de{}, rosenbrock{}, 20u);
    BOOST_CHECK_EQUAL(archi.get_champions_f().size(), 3u);
    BOOST_CHECK_EQUAL(archi.get_champions_x().size(), 3u);
    for (auto i = 0u; i < 3u; ++i) {
        BOOST_CHECK(archi[i].get_population().champion_x() == archi.get_champions_x()[i]);
        BOOST_CHECK(archi[i].get_population().champion_f() == archi.get_champions_f()[i]);
    }
    archi.push_back(de{}, rosenbrock{10}, 20u);
    BOOST_CHECK(archi.get_champions_x()[2].size() == 2u);
    BOOST_CHECK(archi.get_champions_x()[3].size() == 10u);
    archi.push_back(de{}, zdt{}, 20u);
    BOOST_CHECK_THROW(archi.get_champions_f(), std::invalid_argument);
    BOOST_CHECK_THROW(archi.get_champions_x(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(archipelago_status)
{
    flag.store(true);
    archipelago a{10, de{}, population{prob_01{}, 25}};
    BOOST_CHECK(a.status() != evolve_status::busy);
    flag.store(false);
    a.evolve();
    BOOST_CHECK(a.status() == evolve_status::busy);
    flag.store(true);
    a.wait();
    BOOST_CHECK(a.status() == evolve_status::idle);
    flag.store(false);
    a = archipelago{10, de{}, population{rosenbrock{}, 3}};
    a.evolve(10);
    a.evolve(10);
    a.evolve(10);
    a.evolve(10);
    a.wait();
    BOOST_CHECK(a.status() == evolve_status::idle_error);
    // A few idle with errors, one busy.
    a = archipelago{10, de{}, population{rosenbrock{}, 3}};
    a.evolve();
    a.wait();
    flag.store(true);
    a.push_back(de{}, population{prob_01{}, 25});
    flag.store(false);
    a.evolve();
    BOOST_CHECK(a.status() == evolve_status::busy_error);
    flag.store(true);
    a.wait();
    // No busy errors, but only idle errors.
    a = archipelago{10, de{}, population{rosenbrock{}, 3}};
    a.evolve();
    a.wait();
    flag.store(true);
    a.push_back(de{}, population{prob_01{}, 25});
    flag.store(false);
    a[10].evolve();
    BOOST_CHECK(a.status() == evolve_status::busy_error);
    flag.store(true);
    BOOST_CHECK_THROW(a.wait_check(), std::invalid_argument);
    BOOST_CHECK(a.status() == evolve_status::idle);
}

struct pthrower_00 {
    static int counter;
    vector_double fitness(const vector_double &) const
    {
        if (counter >= 50) {
            throw std::invalid_argument("");
        }
        ++counter;
        return {0.};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
};

int pthrower_00::counter = 0;

// Small test about proper cleanup when throwing from the ctor.
BOOST_AUTO_TEST_CASE(archipelago_throw_on_ctor)
{
    BOOST_CHECK_THROW((archipelago{100u, de{}, pthrower_00{}, 1u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(archipelago_bfe_ctors)
{
    archipelago archi00{100, de{}, rosenbrock{20}, bfe{}, 100u, 42u};
    BOOST_CHECK(archi00.size() == 100u);
    for (const auto &isl : archi00) {
        BOOST_CHECK(isl.get_algorithm().is<de>());
        auto pop = isl.get_population();
        BOOST_CHECK(pop.get_problem().is<rosenbrock>());
        BOOST_CHECK(pop.size() == 100u);
        BOOST_CHECK(pop.get_problem().get_fevals() == 100u);
        for (auto i = 0u; i < 100u; ++i) {
            BOOST_CHECK(pop.get_f()[i] == pop.get_problem().fitness(pop.get_x()[i]));
        }
    }

    // With udbfe.
    archi00 = archipelago{100, de{}, rosenbrock{20}, thread_bfe{}, 100u, 42u};
    BOOST_CHECK(archi00.size() == 100u);
    for (const auto &isl : archi00) {
        BOOST_CHECK(isl.get_algorithm().is<de>());
        auto pop = isl.get_population();
        BOOST_CHECK(pop.get_problem().is<rosenbrock>());
        BOOST_CHECK(pop.size() == 100u);
        BOOST_CHECK(pop.get_problem().get_fevals() == 100u);
        for (auto i = 0u; i < 100u; ++i) {
            BOOST_CHECK(pop.get_f()[i] == pop.get_problem().fitness(pop.get_x()[i]));
        }
    }

    archi00 = archipelago{100, thread_island{}, de{}, rosenbrock{20}, bfe{}, 100u, 42u};
    BOOST_CHECK(archi00.size() == 100u);
    for (const auto &isl : archi00) {
        BOOST_CHECK(isl.is<thread_island>());
        BOOST_CHECK(isl.get_algorithm().is<de>());
        auto pop = isl.get_population();
        BOOST_CHECK(pop.get_problem().is<rosenbrock>());
        BOOST_CHECK(pop.size() == 100u);
        BOOST_CHECK(pop.get_problem().get_fevals() == 100u);
        for (auto i = 0u; i < 100u; ++i) {
            BOOST_CHECK(pop.get_f()[i] == pop.get_problem().fitness(pop.get_x()[i]));
        }
    }

    // With udbfe.
    archi00 = archipelago{100, thread_island{}, de{}, rosenbrock{20}, thread_bfe{}, 100u, 42u};
    BOOST_CHECK(archi00.size() == 100u);
    for (const auto &isl : archi00) {
        BOOST_CHECK(isl.is<thread_island>());
        BOOST_CHECK(isl.get_algorithm().is<de>());
        auto pop = isl.get_population();
        BOOST_CHECK(pop.get_problem().is<rosenbrock>());
        BOOST_CHECK(pop.size() == 100u);
        BOOST_CHECK(pop.get_problem().get_fevals() == 100u);
        for (auto i = 0u; i < 100u; ++i) {
            BOOST_CHECK(pop.get_f()[i] == pop.get_problem().fitness(pop.get_x()[i]));
        }
    }

    // Try also a ctor with bfe argument but without seed argument.
    archi00 = archipelago{100, de{}, rosenbrock{20}, bfe{}, 100u};
    BOOST_CHECK(archi00.size() == 100u);
    for (const auto &isl : archi00) {
        BOOST_CHECK(isl.get_algorithm().is<de>());
        auto pop = isl.get_population();
        BOOST_CHECK(pop.get_problem().is<rosenbrock>());
        BOOST_CHECK(pop.size() == 100u);
        BOOST_CHECK(pop.get_problem().get_fevals() == 100u);
        for (auto i = 0u; i < 100u; ++i) {
            BOOST_CHECK(pop.get_f()[i] == pop.get_problem().fitness(pop.get_x()[i]));
        }
    }

    // With udbfe.
    archi00 = archipelago{100, de{}, rosenbrock{20}, thread_bfe{}, 100u};
    BOOST_CHECK(archi00.size() == 100u);
    for (const auto &isl : archi00) {
        BOOST_CHECK(isl.get_algorithm().is<de>());
        auto pop = isl.get_population();
        BOOST_CHECK(pop.get_problem().is<rosenbrock>());
        BOOST_CHECK(pop.size() == 100u);
        BOOST_CHECK(pop.get_problem().get_fevals() == 100u);
        for (auto i = 0u; i < 100u; ++i) {
            BOOST_CHECK(pop.get_f()[i] == pop.get_problem().fitness(pop.get_x()[i]));
        }
    }

    // Try also a ctor with UDI, bfe argument but without seed argument.
    archi00 = archipelago{100, thread_island{}, de{}, rosenbrock{20}, bfe{}, 100u};
    BOOST_CHECK(archi00.size() == 100u);
    for (const auto &isl : archi00) {
        BOOST_CHECK(isl.is<thread_island>());
        BOOST_CHECK(isl.get_algorithm().is<de>());
        auto pop = isl.get_population();
        BOOST_CHECK(pop.get_problem().is<rosenbrock>());
        BOOST_CHECK(pop.size() == 100u);
        BOOST_CHECK(pop.get_problem().get_fevals() == 100u);
        for (auto i = 0u; i < 100u; ++i) {
            BOOST_CHECK(pop.get_f()[i] == pop.get_problem().fitness(pop.get_x()[i]));
        }
    }

    // With udbfe.
    archi00 = archipelago{100, thread_island{}, de{}, rosenbrock{20}, thread_bfe{}, 100u};
    BOOST_CHECK(archi00.size() == 100u);
    for (const auto &isl : archi00) {
        BOOST_CHECK(isl.is<thread_island>());
        BOOST_CHECK(isl.get_algorithm().is<de>());
        auto pop = isl.get_population();
        BOOST_CHECK(pop.get_problem().is<rosenbrock>());
        BOOST_CHECK(pop.size() == 100u);
        BOOST_CHECK(pop.get_problem().get_fevals() == 100u);
        for (auto i = 0u; i < 100u; ++i) {
            BOOST_CHECK(pop.get_f()[i] == pop.get_problem().fitness(pop.get_x()[i]));
        }
    }
}

// Test case for a bug in multi-objective migration in pagmo 2.11.
BOOST_AUTO_TEST_CASE(archipelago_mo_migration_bug)
{
    archipelago a{ring{}, 10u, nsga2{100}, dtlz{2, 50}, 100u};

    a.evolve(4);
    BOOST_CHECK_NO_THROW(a.wait_check());
}

BOOST_AUTO_TEST_CASE(archipelago_set_migrants_db)
{
    archipelago a{ring{}, 10u, nsga2{10}, dtlz{2, 50}, 100u};
    a.evolve(4);
    a.wait_check();

    BOOST_CHECK(a.get_migrants_db().size() == 10u);

    // Set a db of empty groups.
    a.set_migrants_db(archipelago::migrants_db_t(10u));
    BOOST_CHECK(a.get_migrants_db().size() == 10u);
    for (const auto &ig : a.get_migrants_db()) {
        BOOST_CHECK(std::get<0>(ig).empty());
        BOOST_CHECK(std::get<1>(ig).empty());
        BOOST_CHECK(std::get<2>(ig).empty());
    }

    // Try putting in a correct migrants db.
    a = archipelago{ring{}, 10u, nsga2{10}, dtlz{2, 50}, 100u};
    archipelago::migrants_db_t new_db(10u);
    std::get<0>(new_db[0]).push_back(42);
    std::get<1>(new_db[0]).push_back(vector_double(50u));
    std::get<2>(new_db[0]).push_back(vector_double(3u));
    a.set_migrants_db(new_db);

    a.evolve(4);
    BOOST_CHECK_NO_THROW(a.wait_check());
}
