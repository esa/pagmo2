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

#define BOOST_TEST_MODULE algorithm_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

// Complete algorithm stochastic
struct al_01 {
    al_01(){};
    population evolve(population pop) const
    {
        return pop;
    };
    std::string get_name() const
    {
        return "name";
    };
    std::string get_extra_info() const
    {
        return "\tSeed: " + std::to_string(m_seed) + "\n\tVerbosity: " + std::to_string(m_verbosity);
    };
    void set_seed(unsigned int seed)
    {
        m_seed = seed;
    };
    void set_verbosity(unsigned int level)
    {
        m_verbosity = level;
    };
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_seed, m_verbosity);
    }
    unsigned int m_seed = 0u;
    unsigned int m_verbosity = 0u;
};
PAGMO_REGISTER_ALGORITHM(al_01)

// Minimal algorithm deterministic
struct al_02 {
    al_02(){};
    population evolve(population pop) const
    {
        return pop;
    };
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};
PAGMO_REGISTER_ALGORITHM(al_02)

BOOST_AUTO_TEST_CASE(algorithm_construction_test)
{
    // We construct two different algorithms. One having all the
    // mandatory and optional methods implemented, the other
    // having only the mandatory methods implemented
    algorithm algo_full{al_01{}};
    algorithm algo_minimal{al_02{}};
    // We test that the optional methods are appropiately detected in the full case
    BOOST_CHECK(algo_full.has_set_seed() == true);
    BOOST_CHECK(algo_full.has_set_verbosity() == true);
    BOOST_CHECK_NO_THROW(algo_full.set_seed(1u));
    BOOST_CHECK_NO_THROW(algo_full.set_verbosity(1u));
    // And in the minimal case
    BOOST_CHECK(algo_minimal.has_set_seed() == false);
    BOOST_CHECK(algo_minimal.has_set_verbosity() == false);
    BOOST_CHECK_THROW(algo_minimal.set_seed(1u), not_implemented_error);
    BOOST_CHECK_THROW(algo_minimal.set_verbosity(1u), not_implemented_error);
    // We check that at construction the name has been assigned
    BOOST_CHECK(algo_full.get_name() == "name");
    BOOST_CHECK(algo_minimal.get_name().find("al_02") != std::string::npos);
    // Default constructor.
    algorithm a0;
    BOOST_CHECK((a0.extract<null_algorithm>() != nullptr));
    // Check copy semantics.
    algorithm a1{a0};
    BOOST_CHECK((a0.extract<null_algorithm>() != nullptr));
    BOOST_CHECK((a1.extract<null_algorithm>() != nullptr));
    algorithm a2{al_01{}};
    a2 = a1;
    BOOST_CHECK((a2.extract<null_algorithm>() != nullptr));
    BOOST_CHECK((a1.extract<null_algorithm>() != nullptr));
    // Move semantics.
    algorithm a3{std::move(a0)};
    BOOST_CHECK((a3.extract<null_algorithm>() != nullptr));
    algorithm a4{al_01{}};
    a4 = std::move(a2);
    BOOST_CHECK((a4.extract<null_algorithm>() != nullptr));
    // Check we can revive moved-from objects.
    a0 = a4;
    BOOST_CHECK((a0.extract<null_algorithm>() != nullptr));
    a2 = std::move(a4);
    BOOST_CHECK((a2.extract<null_algorithm>() != nullptr));

    // Check the is_uda type trait.
    BOOST_CHECK(is_uda<al_01>::value);
    BOOST_CHECK(is_uda<null_algorithm>::value);
    BOOST_CHECK(!is_uda<al_01 &>::value);
    BOOST_CHECK(!is_uda<const al_01>::value);
    BOOST_CHECK(!is_uda<int>::value);
    BOOST_CHECK(!is_uda<void>::value);
    BOOST_CHECK(!is_uda<std::string>::value);
    BOOST_CHECK((std::is_constructible<algorithm, al_01>::value));
    BOOST_CHECK((std::is_constructible<algorithm, null_algorithm>::value));
    BOOST_CHECK((std::is_constructible<algorithm, al_01 &>::value));
    BOOST_CHECK((std::is_constructible<algorithm, const null_algorithm &>::value));
    BOOST_CHECK((std::is_constructible<algorithm, al_01 &&>::value));
    BOOST_CHECK((!std::is_constructible<algorithm, int>::value));
    BOOST_CHECK((!std::is_constructible<algorithm, std::string>::value));
}

BOOST_AUTO_TEST_CASE(algorithm_copy_constructor_test)
{
    // We check the copy constructor
    algorithm algo{al_01{}};

    // We set the seed and verbosity so that the default values are changed
    algo.set_seed(1u);
    algo.set_verbosity(1u);

    // We call the copy constructor
    algorithm algo_copy(algo);
    // We extract the user algorithm
    auto a1 = algo.extract<al_01>();
    auto a2 = algo_copy.extract<al_01>();

    // 1 - We check the resources pointed to by m_ptr have different address
    BOOST_CHECK(a1 != 0);
    BOOST_CHECK(a2 != 0);
    BOOST_CHECK(a1 != a2);
    // 2 - We check that the other members are copied
    BOOST_CHECK(algo.get_name() == algo_copy.get_name());
    BOOST_CHECK(algo.has_set_seed() == algo_copy.has_set_seed());
    BOOST_CHECK(algo.has_set_verbosity() == algo_copy.has_set_verbosity());
}

BOOST_AUTO_TEST_CASE(algorithm_move_constructor_test)
{
    // We instantiate an algorithm
    algorithm algo{al_01{}};

    // We set the seed and verbosity so that the default values are changed
    algo.set_seed(1u);
    algo.set_verbosity(1u);

    // We store a streaming representation of the object
    auto algo_string = boost::lexical_cast<std::string>(algo);
    // We get the memory address where the user algo is stored
    auto a1 = algo.extract<al_01>();
    // We call the move constructor
    algorithm moved_algo(std::move(algo));
    // We get the memory address where the user algo is stored
    auto a2 = moved_algo.extract<al_01>();
    // And the string representation of the moved algo
    auto moved_algo_string = boost::lexical_cast<std::string>(moved_algo);
    // 1 - We check the resource pointed by m_ptr has been moved from algo to moved_algo
    BOOST_CHECK(a1 == a2);
    // 2 - We check that the two string representations are identical
    BOOST_CHECK(algo_string == moved_algo_string);
}

// Algorithm with overrides
struct al_03 {
    al_03(){};
    population evolve(population pop) const
    {
        return pop;
    };
    std::string get_name() const
    {
        return "name";
    };
    std::string get_extra_info() const
    {
        return "\tSeed: " + std::to_string(m_seed) + "\n\tVerbosity: " + std::to_string(m_verbosity);
    };
    void set_seed(unsigned int seed)
    {
        m_seed = seed;
    };
    void set_verbosity(unsigned int level)
    {
        m_verbosity = level;
    };
    unsigned int m_seed = 0u;
    unsigned int m_verbosity = 0u;
    bool has_set_seed() const
    {
        return false;
    };
    bool has_set_verbosity() const
    {
        return false;
    };
};

BOOST_AUTO_TEST_CASE(algorithm_override_mechanics_test)
{
    algorithm algo{al_03{}};
    BOOST_CHECK(algo.has_set_seed() == false);
    BOOST_CHECK(algo.has_set_verbosity() == false);
}

BOOST_AUTO_TEST_CASE(algorithm_move_assignment_test)
{
    // We instantiate an algorithm
    algorithm algo{al_01{}};

    // We set the seed and verbosity so that the default values are changed
    algo.set_seed(1u);
    algo.set_verbosity(1u);

    // We store a streaming representation of the object
    auto algo_string = boost::lexical_cast<std::string>(algo);
    // We get the memory address where the user algo is stored
    auto a1 = algo.extract<al_01>();
    // We call the move assignment
    algorithm moved_algo{null_algorithm{}};
    moved_algo = std::move(algo);
    // We get the memory address where the user algo is stored
    auto a2 = moved_algo.extract<al_01>();
    // And the string representation of the moved algo
    auto moved_algo_string = boost::lexical_cast<std::string>(moved_algo);
    // 1 - We check the resource pointed by m_ptr has been moved from algo to moved_algo
    BOOST_CHECK(a1 == a2);
    // 2 - We check that the two string representations are identical
    BOOST_CHECK(algo_string == moved_algo_string);
}

BOOST_AUTO_TEST_CASE(algorithm_copy_assignment_test)
{
    // We check the copy constructor
    algorithm algo{al_01{}};

    // We set the seed and verbosity so that the default values are changed
    algo.set_seed(1u);
    algo.set_verbosity(1u);

    // We call the copy assignment opeator
    algorithm algo_copy{null_algorithm{}};
    algo_copy = algo;
    // We extract the user algorithm
    auto a1 = algo.extract<al_01>();
    auto a2 = algo_copy.extract<al_01>();

    // 1 - We check the resources pointed to by m_ptr have different address
    BOOST_CHECK(a1 != 0);
    BOOST_CHECK(a2 != 0);
    BOOST_CHECK(a1 != a2);
    // 2 - We check that the other members are copied
    BOOST_CHECK(algo.get_name() == algo_copy.get_name());
    BOOST_CHECK(algo.has_set_seed() == algo_copy.has_set_seed());
    BOOST_CHECK(algo.has_set_verbosity() == algo_copy.has_set_verbosity());
}

BOOST_AUTO_TEST_CASE(algorithm_extract_is_test)
{
    algorithm algo{al_01{}};
    algo.set_seed(1u);
    algo.set_verbosity(1u);
    auto user_algo = algo.extract<al_01>();

    // We check thet we can access to public data members
    BOOST_CHECK(user_algo->m_seed == 1u);
    BOOST_CHECK(user_algo->m_verbosity == 1u);

    // We check that a non succesfull cast returns a null pointer
    BOOST_CHECK(!algo.extract<al_02>());

    // We check the is method
    BOOST_CHECK(algo.is<al_01>());
    BOOST_CHECK(!algo.is<al_02>());
}

BOOST_AUTO_TEST_CASE(algorithm_evolve_test)
{
    algorithm algo{al_01{}};
    population pop{problem{rosenbrock{5}}, 2u};
    population pop_out = algo.evolve(pop);
    // We test that the evolve is called and does what
    // its supposed to, in this case return the same population
    BOOST_CHECK(pop.size() == pop_out.size());
    BOOST_CHECK(pop.get_x() == pop_out.get_x());
    BOOST_CHECK(pop.get_f() == pop_out.get_f());
    BOOST_CHECK(pop.get_ID() == pop_out.get_ID());
}

BOOST_AUTO_TEST_CASE(algorithm_setters_test)
{
    algorithm algo{al_01{}};
    algo.set_seed(32u);
    BOOST_CHECK(algo.extract<al_01>()->m_seed == 32u);
    algo.set_verbosity(32u);
    BOOST_CHECK(algo.extract<al_01>()->m_verbosity == 32u);
}

BOOST_AUTO_TEST_CASE(algorithm_has_test)
{
    algorithm algo{al_01{}};
    BOOST_CHECK(algo.has_set_seed() == true);
    BOOST_CHECK(algo.has_set_seed() == algo.is_stochastic());
    BOOST_CHECK(algo.has_set_verbosity() == true);
}

BOOST_AUTO_TEST_CASE(algorithm_getters_test)
{
    {
        algorithm algo{al_01{}};
        BOOST_CHECK(algo.get_name() == "name");
        BOOST_CHECK(algo.get_extra_info().find("Seed") != std::string::npos);
    }
    algorithm algo{al_02{}};
    BOOST_CHECK(algo.get_name().find("al_02") != std::string::npos);
    BOOST_CHECK(algo.get_extra_info().find("") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(algorithm_serialization_test)
{
    // Instantiate an algorithm
    algorithm algo{al_01{}};
    // Change its state
    algo.set_seed(2u);
    algo.set_verbosity(2u);
    // Store the string representation.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(algo);
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(algo);
    }
    // Create a new algorithm object
    auto algo2 = algorithm{al_02{}};
    boost::lexical_cast<std::string>(algo2); // triggers the streaming operator for a deterministic algo
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(algo2);
    }
    auto after = boost::lexical_cast<std::string>(algo2);
    BOOST_CHECK_EQUAL(before, after);
    // Check explicitly that the properties of base_p where restored as well.
    BOOST_CHECK_EQUAL(algo.extract<al_01>()->m_seed, algo2.extract<al_01>()->m_seed);
    BOOST_CHECK_EQUAL(algo.extract<al_01>()->m_verbosity, algo2.extract<al_01>()->m_verbosity);
}

BOOST_AUTO_TEST_CASE(null_algorithm_construction_and_evolve)
{
    // Trivial checks
    null_algorithm user_algo{};
    // Evolve check (population does not change)
    rosenbrock user_prob{};
    population pop(user_prob, 20u);
    auto evolved_pop = user_algo.evolve(pop);
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        BOOST_CHECK(pop.get_x()[i] == evolved_pop.get_x()[i]);
        BOOST_CHECK(pop.get_f()[i] == evolved_pop.get_f()[i]);
        BOOST_CHECK(pop.get_ID()[i] == evolved_pop.get_ID()[i]);
    }
}

BOOST_AUTO_TEST_CASE(serialization_test)
{
    algorithm algo{null_algorithm{}};
    BOOST_CHECK_EQUAL(algo.get_name(), "Null algorithm");
    std::stringstream ss;
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(algo);
    }
    algo = algorithm{de{}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(algo);
    }
}

BOOST_AUTO_TEST_CASE(extract_test)
{
    algorithm p{null_algorithm{}};
    BOOST_CHECK(p.is<null_algorithm>());
    BOOST_CHECK((std::is_same<null_algorithm *, decltype(p.extract<null_algorithm>())>::value));
    BOOST_CHECK((std::is_same<null_algorithm const *,
                              decltype(static_cast<const algorithm &>(p).extract<null_algorithm>())>::value));
    BOOST_CHECK(p.extract<null_algorithm>() != nullptr);
    BOOST_CHECK(static_cast<const algorithm &>(p).extract<null_algorithm>() != nullptr);
}

struct ts1 {
    population evolve(const population &) const
    {
        return population{};
    }
};

struct ts2 {
    population evolve(const population &) const
    {
        return population{};
    }
    thread_safety get_thread_safety() const
    {
        return thread_safety::none;
    }
};

struct ts3 {
    population evolve(const population &) const
    {
        return population{};
    }
    int get_thread_safety() const
    {
        return 2;
    }
};

BOOST_AUTO_TEST_CASE(thread_safety_test)
{
    BOOST_CHECK(algorithm{null_algorithm{}}.get_thread_safety() == thread_safety::basic);
    BOOST_CHECK(algorithm{ts1{}}.get_thread_safety() == thread_safety::basic);
    BOOST_CHECK(algorithm{ts2{}}.get_thread_safety() == thread_safety::none);
    BOOST_CHECK(algorithm{ts3{}}.get_thread_safety() == thread_safety::basic);
}
