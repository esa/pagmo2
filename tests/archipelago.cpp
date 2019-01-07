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

#if defined(_MSC_VER)

#define _SCL_SECURE_NO_WARNINGS

#endif

#define BOOST_TEST_MODULE archipelago_test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <atomic>
#include <boost/lexical_cast.hpp>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pagmo/algorithms/de.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(archipelago_construction)
{
    // Make the test deterministic.
    random_device::set_seed(123u);

    using size_type = archipelago::size_type;
    archipelago archi;
    BOOST_CHECK(archi.size() == 0u);
    archipelago archi2(0u, de{}, rosenbrock{}, 10u);
    BOOST_CHECK(archi2.size() == 0u);
    archipelago archi3(5u, de{}, rosenbrock{}, 10u);
    BOOST_CHECK(archi3.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi3[i].status() != evolve_status::busy);
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
    }
    archi3 = archipelago{5u, thread_island{}, de{}, rosenbrock{}, 10u};
    BOOST_CHECK(archi3.size() == 5u);
    std::vector<unsigned> seeds;
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi3[i].status() != evolve_status::busy);
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
        seeds.push_back(archi3[i].get_population().get_seed());
    }
    // Check seeds are different.
    std::sort(seeds.begin(), seeds.end());
    BOOST_CHECK(std::unique(seeds.begin(), seeds.end()) == seeds.end());
    archi3 = archipelago{5u, thread_island{}, de{}, population{rosenbrock{}, 10u}};
    BOOST_CHECK(archi3.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi3[i].status() != evolve_status::busy);
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
    }
    archi3 = archipelago{5u, thread_island{}, de{}, population{rosenbrock{}, 10u, 123u}};
    BOOST_CHECK(archi3.size() == 5u);
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi3[i].status() != evolve_status::busy);
        BOOST_CHECK(archi3[i].get_algorithm().is<de>());
        BOOST_CHECK(archi3[i].get_population().size() == 10u);
        BOOST_CHECK(archi3[i].get_population().get_problem().is<rosenbrock>());
    }
    // A couple of tests for the constructor which contains a seed argument.
    archipelago archi3a{5u, thread_island{}, de{}, rosenbrock{}, 10u, 123u};
    seeds.clear();
    std::transform(archi3a.begin(), archi3a.end(), std::back_inserter(seeds),
                   [](const island &isl) { return isl.get_population().get_seed(); });
    std::sort(seeds.begin(), seeds.end());
    BOOST_CHECK(std::unique(seeds.begin(), seeds.end()) == seeds.end());
    std::vector<unsigned> seeds2;
    archipelago archi3b{5u, de{}, rosenbrock{}, 10u, 123u};
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
    for (size_type i = 0; i < 5u; ++i) {
        BOOST_CHECK(archi4[i].status() != evolve_status::busy);
        BOOST_CHECK(archi4[i].get_algorithm().is<de>());
        BOOST_CHECK(archi4[i].get_population().size() == 10u);
        BOOST_CHECK(archi4[i].get_population().get_problem().is<rosenbrock>());
    }
    archi4.evolve(10);
    auto archi5 = archi4;
    BOOST_CHECK(archi5.size() == 5u);
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
    archi4 = archi4;
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
}

BOOST_AUTO_TEST_CASE(archipelago_serialization)
{
    archipelago a{10, de{}, population{rosenbrock{}, 25}};
    a.evolve();
    a.wait_check();
    BOOST_CHECK(a.status() == evolve_status::idle);
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(a);
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(a);
    }
    a = archipelago{10, de{}, population{rosenbrock{}, 25}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(a);
    }
    auto after = boost::lexical_cast<std::string>(a);
    BOOST_CHECK_EQUAL(before, after);
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
