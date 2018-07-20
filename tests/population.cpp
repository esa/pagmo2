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

#define BOOST_TEST_MODULE population_test

#include <boost/lexical_cast.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

static inline std::string pop_to_string(const population &pop)
{
    std::stringstream ss;
    ss << pop;
    return ss.str();
}

BOOST_AUTO_TEST_CASE(population_construction_test)
{
    unsigned int seed = 123;
    population pop1{};
    population pop2{problem{zdt{1, 5}}, 2, seed};
    population pop3{problem{zdt{2, 5}}, 2, seed};

    // We check that the number of individuals is as expected
    BOOST_CHECK(pop1.size() == 0u);
    BOOST_CHECK(pop2.size() == 2u);
    BOOST_CHECK(pop3.size() == 2u);
    // We check population's individual chromosomes and IDs are the same
    // as the random seed was (and the problem dimension), while
    // fitness vectors were different as the problem is
    BOOST_CHECK(pop2.get_ID() == pop3.get_ID());
    BOOST_CHECK(pop2.get_x() == pop3.get_x());
    BOOST_CHECK(pop2.get_f() != pop3.get_f());
    // We check that the seed has been set correctly
    BOOST_CHECK(pop2.get_seed() == seed);

    // We test the generic constructor
    population pop4{zdt{2, 5}, 2, seed};
    BOOST_CHECK(pop4.get_ID() == pop3.get_ID());
    BOOST_CHECK(pop4.get_x() == pop3.get_x());
    BOOST_CHECK(pop4.get_f() == pop3.get_f());
    population pop5{zdt{1, 5}, 2, seed};
    BOOST_CHECK(pop2.get_ID() == pop5.get_ID());
    BOOST_CHECK(pop2.get_x() == pop5.get_x());
    BOOST_CHECK(pop2.get_f() == pop5.get_f());

    // Check copy/move semantics.
    population pop_a{problem{zdt{2, 5}}, 2, 20};
    population pop_b{pop_a};
    BOOST_CHECK_EQUAL(pop_to_string(pop_a), pop_to_string(pop_b));
    population pop_c;
    pop_c = pop_b;
    BOOST_CHECK_EQUAL(pop_to_string(pop_a), pop_to_string(pop_c));
    population pop_d{std::move(pop_c)};
    BOOST_CHECK_EQUAL(pop_to_string(pop_a), pop_to_string(pop_d));
    population pop_e;
    pop_e = std::move(pop_b);
    BOOST_CHECK_EQUAL(pop_to_string(pop_a), pop_to_string(pop_e));
    // Try to revive moved-from objects.
    pop_c = pop_e;
    BOOST_CHECK_EQUAL(pop_to_string(pop_a), pop_to_string(pop_c));
    pop_b = std::move(pop_e);
    BOOST_CHECK_EQUAL(pop_to_string(pop_a), pop_to_string(pop_b));

    // Self assignments.
    pop_a = pop_b;
    pop_a = pop_a;
    BOOST_CHECK_EQUAL(pop_to_string(pop_a), pop_to_string(pop_b));
#if !defined(__clang__)
    pop_a = std::move(pop_a);
    BOOST_CHECK_EQUAL(pop_to_string(pop_a), pop_to_string(pop_b));
#endif

    // Check constructability.
    BOOST_CHECK((!std::is_constructible<population, int>::value));
    BOOST_CHECK((!std::is_constructible<population, int &>::value));
    BOOST_CHECK((!std::is_constructible<population, const int &>::value));
    BOOST_CHECK((!std::is_constructible<population, std::string>::value));
    BOOST_CHECK((std::is_constructible<population, null_problem>::value));
    BOOST_CHECK((std::is_constructible<population, null_problem &>::value));
    BOOST_CHECK((std::is_constructible<population, null_problem &&>::value));
    BOOST_CHECK((std::is_constructible<population, const null_problem &>::value));
    BOOST_CHECK((std::is_constructible<population, const null_problem>::value));
    BOOST_CHECK((std::is_constructible<population, problem>::value));
    BOOST_CHECK((std::is_constructible<population, problem &>::value));
    BOOST_CHECK((std::is_constructible<population, problem &&>::value));
    BOOST_CHECK((std::is_constructible<population, const problem &>::value));
    BOOST_CHECK((std::is_constructible<population, const problem>::value));
}

BOOST_AUTO_TEST_CASE(population_copy_constructor_test)
{
    population pop1{problem{rosenbrock{5}}, 10u};
    population pop2(pop1);
    BOOST_CHECK(pop2.get_ID() == pop1.get_ID());
    BOOST_CHECK(pop2.get_x() == pop1.get_x());
    BOOST_CHECK(pop2.get_f() == pop1.get_f());
}

struct malformed {
    vector_double fitness(const vector_double &) const
    {
        return {0.5};
    }
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
};

BOOST_AUTO_TEST_CASE(population_push_back_test)
{
    // Create an empty population
    population pop{problem{zdt{1u, 30u}}};
    // We fill it with a few individuals and check the size growth
    for (unsigned int i = 0u; i < 5u; ++i) {
        BOOST_CHECK(pop.size() == i);
        BOOST_CHECK(pop.get_f().size() == i);
        BOOST_CHECK(pop.get_x().size() == i);
        BOOST_CHECK(pop.get_ID().size() == i);
        pop.push_back(vector_double(30u, 0.5));
    }
    // We check the fitness counter
    BOOST_CHECK(pop.get_problem().get_fevals() == 5u);
    // We check important undefined throws
    // 1 - Cannot push back the wrong decision vector dimension
    BOOST_CHECK_THROW(pop.push_back(vector_double(28u, 0.5)), std::invalid_argument);
    // 2 - Malformed problem. The user declares 2 objectives but returns something else
    population pop2{problem{malformed{}}};
    BOOST_CHECK_THROW(pop2.push_back({1.}), std::invalid_argument);
    // 3 - Consistency checks on the second push_back() overload.
    population pop3{problem{zdt{1u, 30u}}};
    BOOST_CHECK_THROW(pop3.push_back({}, {}), std::invalid_argument);
    BOOST_CHECK_THROW(pop3.push_back(vector_double(30u, 0.5), {}), std::invalid_argument);
    BOOST_CHECK_THROW(pop3.push_back(vector_double(30u, 0.5), {0.}), std::invalid_argument);
    BOOST_CHECK_THROW(pop3.push_back(vector_double(30u, 0.5), {0., 0., 0.}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(population_random_decision_vector_test)
{
    // Create an empty population
    population pop{problem{null_problem{}}};
    auto bounds = pop.get_problem().get_bounds();
    // Generate a random decision_vector
    auto x = pop.random_decision_vector();
    // Check that the decision_vector is indeed within bounds
    for (decltype(x.size()) i = 0u; i < x.size(); ++i) {
        BOOST_CHECK(x[i] < bounds.second[i]);
        BOOST_CHECK(x[i] >= bounds.first[i]);
    }
}

BOOST_AUTO_TEST_CASE(population_best_worst_test)
{
    // Test throw
    {
        population pop{problem{zdt{}}, 2};
        population pop2{problem{null_problem{}}, 0u};
        BOOST_CHECK_THROW(pop.best_idx(), std::invalid_argument);
        BOOST_CHECK_THROW(pop.worst_idx(), std::invalid_argument);
        BOOST_CHECK_THROW(pop2.best_idx(), std::invalid_argument);
        BOOST_CHECK_THROW(pop2.worst_idx(), std::invalid_argument);
    }
    // Test on single objective
    {
        population pop{problem{rosenbrock{2}}};
        pop.push_back({0.5, 0.5});
        pop.push_back(pop.get_problem().extract<rosenbrock>()->best_known());
        BOOST_CHECK(pop.worst_idx() == 0u);
        BOOST_CHECK(pop.best_idx() == 1u);
    }
    // Test on constrained
    {
        population pop{problem{hock_schittkowsky_71{}}};
        pop.push_back({1.5, 1.5, 1.5, 1.5});
        pop.push_back(pop.get_problem().extract<hock_schittkowsky_71>()->best_known());
        BOOST_CHECK(pop.worst_idx(1e-5) == 0u); // tolerance matter here!!!
        BOOST_CHECK(pop.best_idx(1e-5) == 1u);
    }
}

BOOST_AUTO_TEST_CASE(population_setters_test)
{
    population pop{problem{null_problem{}}, 2};
    // Test throw
    BOOST_CHECK_THROW(pop.set_xf(2, {3}, {1, 2, 3}), std::invalid_argument); // index invalid
    BOOST_CHECK_THROW(pop.set_xf(1, {3, 2}, {1}), std::invalid_argument);    // chromosome invalid
    BOOST_CHECK_THROW(pop.set_xf(1, {3}, {1, 2}), std::invalid_argument);    // fitness invalid
    // Test set_xf
    pop.set_xf(0, {3}, {1});
    BOOST_CHECK((pop.get_x()[0] == vector_double{3}));
    BOOST_CHECK((pop.get_f()[0] == vector_double{1}));
    // Test set_x
    pop.set_x(0, {1.2});
    BOOST_CHECK((pop.get_x()[0] == vector_double{1.2}));
    BOOST_CHECK(pop.get_f()[0] == pop.get_problem().fitness({1.2})); // works as counters are marked mutable
}

BOOST_AUTO_TEST_CASE(population_getters_test)
{
    population pop{problem{null_problem{}}, 1, 1234u};
    pop.set_xf(0, {3}, {1});
    // Test
    BOOST_CHECK((pop.get_f()[0] == vector_double{1}));
    BOOST_CHECK(pop.get_seed() == 1234u);
    BOOST_CHECK_NO_THROW(pop.get_ID());
    // Streaming operator is tested to contain the problem stream
    auto pop_string = boost::lexical_cast<std::string>(pop);
    auto prob_string = boost::lexical_cast<std::string>(pop.get_problem());
    BOOST_CHECK(pop_string.find(prob_string) != std::string::npos);
}

BOOST_AUTO_TEST_CASE(population_champion_test)
{
    // Unconstrained case
    {
        population pop{problem{rosenbrock{2u}}};
        // Upon construction of an empty population the Champion is empty
        BOOST_CHECK((pop.champion_x() == vector_double{}));
        BOOST_CHECK((pop.champion_f() == vector_double{}));
        // We push back the origin, in Rosenbrock this has a fitness of 1.
        pop.push_back({0., 0.});
        BOOST_CHECK((pop.champion_x() == vector_double{0., 0.}));
        BOOST_CHECK((pop.champion_f() == vector_double{1.}));
        // We push back .1,.1, in Rosenbrock this has a fitness of 1.62 and thus should not trigger the champion update
        pop.push_back({0.1, 0.1});
        BOOST_CHECK((pop.champion_x() == vector_double{0., 0.}));
        BOOST_CHECK((pop.champion_f() == vector_double{1.}));
        // We push back 0.01,0.01, in Rosenbrock this has a fitness of 0.989901 and thus should trigger the champion
        // update
        pop.push_back({0.01, 0.01});
        BOOST_CHECK((pop.champion_x() == vector_double{0.01, 0.01}));
        BOOST_CHECK_CLOSE(pop.champion_f()[0], 0.989901, 1e-6);
        // We set the chromosome of this last individual to something worse, the champion does not change
        pop.set_x(2u, {0.1, 0.1});
        BOOST_CHECK((pop.champion_x() == vector_double{0.01, 0.01}));
        BOOST_CHECK_CLOSE(pop.champion_f()[0], 0.989901, 1e-6);
        // We set the chromosome of this last individual to something better, the champion does change
        pop.set_xf(2u, {0.123, 0.123}, {0.12});
        BOOST_CHECK((pop.champion_x() == vector_double{0.123, 0.123}));
        BOOST_CHECK((pop.champion_f() == vector_double{0.12}));
    }
    // Constrained case
    {
        population pop{problem{hock_schittkowsky_71{}}};
        // Upon construction of an empty population the Champion is empty
        BOOST_CHECK((pop.champion_x() == vector_double{}));
        BOOST_CHECK((pop.champion_f() == vector_double{}));
        // We push back 1.1,1.1,.. in hock_schittkowsky_71 this has a fitness of [  5.093, -35.16, 23.5359]
        pop.push_back({1.1, 1.1, 1.1, 1.1});
        auto ch = pop.champion_f();
        BOOST_CHECK((pop.champion_x() == vector_double{1.1, 1.1, 1.1, 1.1}));
        // We push back all ones, in hock_schittkowsky_71 this has a fitness of [ 4., -36., 24.] and does not trigger a
        // champion update
        pop.push_back({1., 1., 1., 1.});
        BOOST_CHECK((pop.champion_x() == vector_double{1.1, 1.1, 1.1, 1.1}));
        BOOST_CHECK((pop.champion_f() == ch));
        // We push back 2.1,2.1, in hock_schittkowsky_71 this has a fitness of [29.883 ,-22.36, 5.5519] and triggers a
        // champion update
        pop.push_back({2.1, 2.1, 2.1, 2.1});
        BOOST_CHECK((pop.champion_x() == vector_double{2.1, 2.1, 2.1, 2.1}));
        BOOST_CHECK((pop.champion_f() != ch));
        ch = pop.champion_f();
        // We set the chromosome of this last individual to something worse, the champion does not change
        pop.set_xf(2u, {1.2, 1.3, 1.4, 1.5}, {12., 45., 55.});
        BOOST_CHECK((pop.champion_x() == vector_double{2.1, 2.1, 2.1, 2.1}));
        BOOST_CHECK(pop.champion_f() == ch);
        // We set the chromosome of this last individual to something better, the champion does change
        pop.set_xf(2u, {1.2, 1.3, 1.4, 1.5}, {12., 5., -55.});
        BOOST_CHECK((pop.champion_x() == vector_double{1.2, 1.3, 1.4, 1.5}));
        BOOST_CHECK((pop.champion_f() == vector_double{12., 5., -55.}));
    }
    // We check that requests to the champion cannot be made if the population
    // contains a problem with more than 1 objective or is stochastic
    population pop_mo{problem{zdt{}}, 2u};
    BOOST_CHECK_THROW(pop_mo.champion_f(), std::invalid_argument);
    BOOST_CHECK_THROW(pop_mo.champion_x(), std::invalid_argument);
    population pop_sto{problem{inventory{12u}}, 2u};
    BOOST_CHECK_THROW(pop_sto.champion_f(), std::invalid_argument);
    BOOST_CHECK_THROW(pop_sto.champion_x(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(population_serialization_test)
{
    population pop{problem{null_problem{}}, 30, 1234u};
    // Store the string representation of p.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(pop);
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(pop);
    }
    // Change the content of p before deserializing.
    pop = population{problem{zdt{5, 20u}}, 30};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(pop);
    }
    auto after = boost::lexical_cast<std::string>(pop);
    BOOST_CHECK_EQUAL(before, after);
}

struct minlp {
    minlp(vector_double::size_type nix = 0u)
    {
        m_nix = nix;
    }
    vector_double fitness(const vector_double &x) const
    {
        return {std::sin(x[0] * x[1] * x[2]), x[0] + x[1] + x[2], x[0] * x[1] + x[1] * x[2] - x[0] * x[2]};
    }
    vector_double::size_type get_nobj() const
    {
        return 3u;
    }
    vector_double::size_type get_nix() const
    {
        return m_nix;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{1, -1, 1}, {2, 1, 2}};
    }
    vector_double::size_type m_nix;
};

BOOST_AUTO_TEST_CASE(population_minlp_test)
{
    population pop{problem{minlp{2}}, 30, 1234u};
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        BOOST_CHECK(pop.get_x()[i][2] == std::floor(pop.get_x()[i][2]));
        BOOST_CHECK(pop.get_x()[i][1] == std::floor(pop.get_x()[i][1]));
    }
}

BOOST_AUTO_TEST_CASE(population_cout_test)
{
    population pop{problem{rosenbrock{2u}}};
    population pop_sto{problem{inventory{12u}}, 3u};
    population pop_mo{problem{zdt{}}, 3u};
    BOOST_CHECK_NO_THROW(std::cout << pop);
    BOOST_CHECK_NO_THROW(std::cout << pop_sto);
    BOOST_CHECK_NO_THROW(std::cout << pop_mo);
}