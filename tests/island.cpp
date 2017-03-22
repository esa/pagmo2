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

#define BOOST_TEST_MODULE island_test
#include <boost/test/included/unit_test.hpp>

#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <pagmo/algorithms/de.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/rosenbrock.hpp>

using namespace pagmo;

struct udi_01 {
    void run_evolve(island &) const;
};

struct udi_02 {
    void run_evolve(island &);
};

struct udi_03 {
    void run_evolve(const island &) const;
};

BOOST_AUTO_TEST_CASE(island_type_traits)
{
    BOOST_CHECK(is_udi<thread_island>::value);
    BOOST_CHECK(!is_udi<int>::value);
    BOOST_CHECK(!is_udi<const thread_island>::value);
    BOOST_CHECK(!is_udi<const thread_island &>::value);
    BOOST_CHECK(!is_udi<thread_island &>::value);
    BOOST_CHECK(!is_udi<void>::value);
    BOOST_CHECK(is_udi<udi_01>::value);
    BOOST_CHECK(!is_udi<udi_02>::value);
    BOOST_CHECK(is_udi<udi_03>::value);
}

BOOST_AUTO_TEST_CASE(island_constructors)
{
    // Various constructors.
    island isl;
    BOOST_CHECK(isl.get_algorithm().is<null_algorithm>());
    BOOST_CHECK(isl.get_population().get_problem().is<null_problem>());
    BOOST_CHECK(isl.get_population().size() == 0u);
    auto isl2 = isl;
    BOOST_CHECK(isl2.get_algorithm().is<null_algorithm>());
    BOOST_CHECK(isl2.get_population().get_problem().is<null_problem>());
    BOOST_CHECK(isl2.get_population().size() == 0u);
    island isl3{de{}, population{rosenbrock{}, 25}};
    BOOST_CHECK(isl3.get_algorithm().is<de>());
    BOOST_CHECK(isl3.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl3.get_population().size() == 25u);
    auto isl4 = isl3;
    BOOST_CHECK(isl4.get_algorithm().is<de>());
    BOOST_CHECK(isl4.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl4.get_population().size() == 25u);
    island isl5{thread_island{}, de{}, population{rosenbrock{}, 26}};
    BOOST_CHECK(isl5.get_algorithm().is<de>());
    BOOST_CHECK(isl5.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl5.get_population().size() == 26u);
    island isl6{de{}, rosenbrock{}, 27};
    BOOST_CHECK(isl6.get_algorithm().is<de>());
    BOOST_CHECK(isl6.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl6.get_population().size() == 27u);
    island isl7{de{}, rosenbrock{}, 27, 123};
    BOOST_CHECK(isl7.get_algorithm().is<de>());
    BOOST_CHECK(isl7.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl7.get_population().size() == 27u);
    BOOST_CHECK(isl7.get_population().get_seed() == 123u);
    island isl8{thread_island{}, de{}, rosenbrock{}, 28};
    BOOST_CHECK(isl8.get_algorithm().is<de>());
    BOOST_CHECK(isl8.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl8.get_population().size() == 28u);
    island isl9{thread_island{}, de{}, rosenbrock{}, 29, 124};
    BOOST_CHECK(isl9.get_algorithm().is<de>());
    BOOST_CHECK(isl9.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl9.get_population().size() == 29u);
    BOOST_CHECK(isl9.get_population().get_seed() == 124u);
    island isl10{std::move(isl9)};
    BOOST_CHECK(isl10.get_algorithm().is<de>());
    BOOST_CHECK(isl10.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl10.get_population().size() == 29u);
    BOOST_CHECK(isl10.get_population().get_seed() == 124u);
    // Revive isl9;
    isl9 = island{thread_island{}, de{}, rosenbrock{}, 29, 124};
    BOOST_CHECK(isl9.get_algorithm().is<de>());
    BOOST_CHECK(isl9.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl9.get_population().size() == 29u);
    BOOST_CHECK(isl9.get_population().get_seed() == 124u);
    // Copy assignment.
    isl9 = isl8;
    BOOST_CHECK(isl9.get_algorithm().is<de>());
    BOOST_CHECK(isl9.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl9.get_population().size() == 28u);
    // Self assignment.
    BOOST_CHECK((std::is_same<island &, decltype(isl9 = isl9)>::value));
    isl9 = isl9;
    BOOST_CHECK(isl9.get_algorithm().is<de>());
    BOOST_CHECK(isl9.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl9.get_population().size() == 28u);
#if !defined(__clang__)
    BOOST_CHECK((std::is_same<island &, decltype(isl9 = std::move(isl9))>::value));
    isl9 = std::move(isl9);
    BOOST_CHECK(isl9.get_algorithm().is<de>());
    BOOST_CHECK(isl9.get_population().get_problem().is<rosenbrock>());
    BOOST_CHECK(isl9.get_population().size() == 28u);
#endif
    // Some type-traits.
    BOOST_CHECK((std::is_constructible<island, de, population>::value));
    BOOST_CHECK((std::is_constructible<island, de, problem &&, unsigned>::value));
    BOOST_CHECK((std::is_constructible<island, de, problem &&, unsigned>::value));
    BOOST_CHECK((std::is_constructible<island, const thread_island &, de, problem &&, unsigned>::value));
    BOOST_CHECK((!std::is_constructible<island, double, std::string &&>::value));
}

BOOST_AUTO_TEST_CASE(island_concurrent_access)
{
    island isl{de{}, rosenbrock{}, 27, 123};
    auto thread_func = [&isl]() {
        for (auto i = 0; i < 100; ++i) {
            auto pop = isl.get_population();
            isl.set_population(pop);
            auto algo = isl.get_algorithm();
            isl.set_algorithm(algo);
        }
    };
    std::thread t1(thread_func), t2(thread_func), t3(thread_func), t4(thread_func);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
}

BOOST_AUTO_TEST_CASE(island_evolve)
{
    island isl{de{}, population{rosenbrock{}, 25}};
    isl.evolve(0);
    isl.get();
}