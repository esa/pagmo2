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

#define BOOST_TEST_MODULE mbh_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <cmath>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/mbh.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(mbh_algorithm_construction)
{
    compass_search inner_algo{100u, 0.1, 0.001, 0.7};
    {
        mbh user_algo{inner_algo, 5, 1e-3};
        BOOST_CHECK((user_algo.get_perturb() == vector_double{1e-3}));
        BOOST_CHECK(user_algo.get_verbosity() == 0u);
        BOOST_CHECK((user_algo.get_log() == mbh::log_type{}));
    }
    {
        mbh user_algo{inner_algo, 5, {1e-3, 1e-2, 1e-3, 1e-2}};
        BOOST_CHECK((user_algo.get_perturb() == vector_double{1e-3, 1e-2, 1e-3, 1e-2}));
        BOOST_CHECK(user_algo.get_verbosity() == 0u);
        BOOST_CHECK((user_algo.get_log() == mbh::log_type{}));
    }
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, -2.1}), std::invalid_argument);
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, 3.2}), std::invalid_argument);
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, std::nan("")}), std::invalid_argument);
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, {0.2, 0.1, 0.}}), std::invalid_argument);
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, {0.2, 0.1, -0.12}}), std::invalid_argument);
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, {0.2, 1.1, 0.12}}), std::invalid_argument);
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, {0.2, std::nan(""), 0.12}}), std::invalid_argument);
    BOOST_CHECK_NO_THROW(mbh{});
}

BOOST_AUTO_TEST_CASE(mbh_evolve_test)
{
    // Here we only test that evolution is deterministic if the
    // seed is controlled
    {
        problem prob{hock_schittkowsky_71{}};
        prob.set_c_tol({1e-3, 1e-3});
        population pop1{prob, 5u, 23u};
        population pop2{prob, 5u, 23u};
        population pop3{prob, 5u, 23u};

        mbh user_algo1{compass_search{100u, 0.1, 0.001, 0.7}, 5u, 0.1, 23u};
        user_algo1.set_verbosity(1u);
        pop1 = user_algo1.evolve(pop1);

        BOOST_CHECK(user_algo1.get_log().size() > 0u);

        mbh user_algo2{compass_search{100u, 0.1, 0.001, 0.7}, 5u, 0.1, 23u};
        user_algo2.set_verbosity(1u);
        pop2 = user_algo2.evolve(pop2);

        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

        user_algo2.set_seed(23u);
        pop3 = user_algo2.evolve(pop3);

        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());
    }
    // We then check that the evolve throws if called on unsuitable problems
    {
        mbh user_algo{compass_search{100u, 0.1, 0.001, 0.7}, 5u, 0.1, 23u};
        BOOST_CHECK_THROW(user_algo.evolve(population{problem{zdt{}}, 15u}), std::invalid_argument);
    }
    {
        mbh user_algo{compass_search{100u, 0.1, 0.001, 0.7}, 5u, 0.1, 23u};
        BOOST_CHECK_THROW(user_algo.evolve(population{problem{inventory{}}, 15u}), std::invalid_argument);
    }
    // And that it throws if called with a wrong dimension of the perturbation vector
    {
        mbh user_algo{compass_search{100u, 0.1, 0.001, 0.7}, 5u, {1e-3, 1e-2}, 23u};
        BOOST_CHECK_THROW(user_algo.evolve(population{problem{hock_schittkowsky_71{}}, 15u}), std::invalid_argument);
    }
    // And a clean exit for 0 generations
    problem prob{hock_schittkowsky_71{}};
    population pop{prob, 10u};
    BOOST_CHECK((mbh{compass_search{100u, 0.1, 0.001, 0.7}, 0u, {1e-3, 1e-2}, 23u}.evolve(pop).get_x()[0])
                == (pop.get_x()[0]));
}

BOOST_AUTO_TEST_CASE(mbh_setters_getters_test)
{
    mbh user_algo{compass_search{100u, 0.1, 0.001, 0.7}, 5u, {1e-3, 1e-2}, 23u};
    user_algo.set_verbosity(23u);
    BOOST_CHECK(user_algo.get_verbosity() == 23u);
    user_algo.set_seed(23u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    user_algo.set_perturb({0.1, 0.2, 0.3, 0.4});
    BOOST_CHECK((user_algo.get_perturb() == vector_double{0.1, 0.2, 0.3, 0.4}));
    BOOST_CHECK_THROW(user_algo.set_perturb({0.1, std::nan(""), 0.3, 0.4}), std::invalid_argument);
    BOOST_CHECK_THROW(user_algo.set_perturb({0.1, -0.2, 0.3, 0.4}), std::invalid_argument);
    BOOST_CHECK_THROW(user_algo.set_perturb({0.1, 2.3, 0.3, 0.4}), std::invalid_argument);
    BOOST_CHECK(user_algo.get_name().find("Monotonic Basin Hopping") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Inner algorithm extra info") != std::string::npos);
    BOOST_CHECK_NO_THROW(user_algo.get_log());
}

BOOST_AUTO_TEST_CASE(mbh_serialization_test)
{
    // Make one evolution
    problem prob{hock_schittkowsky_71{}};
    population pop{prob, 10u, 23u};
    algorithm algo{mbh{compass_search{100u, 0.1, 0.001, 0.7}, 5u, 1e-3, 23u}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<mbh>()->get_log();
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(algo);
    }
    // Change the content of p before deserializing.
    algo = algorithm{null_algorithm{}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(algo);
    }
    auto after_text = boost::lexical_cast<std::string>(algo);
    auto after_log = algo.extract<mbh>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    // BOOST_CHECK(before_log == after_log); // This fails because of floating point problems when using JSON and cereal
    // so we implement a close check
    BOOST_CHECK(before_log.size() > 0u);
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<1>(before_log[i]), std::get<1>(after_log[i]), 1e-8);
        BOOST_CHECK_EQUAL(std::get<2>(before_log[i]), std::get<2>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<3>(before_log[i]), std::get<3>(after_log[i]), 1e-8);
        BOOST_CHECK_EQUAL(std::get<4>(before_log[i]), std::get<4>(after_log[i]));
    }
}

struct ts1 {
    population evolve(population pop) const
    {
        return pop;
    }
};

struct ts2 {
    population evolve(population pop) const
    {
        return pop;
    }
    thread_safety get_thread_safety() const
    {
        return thread_safety::none;
    }
};

struct ts3 {
    population evolve(population pop) const
    {
        return pop;
    }
    thread_safety get_thread_safety()
    {
        return thread_safety::none;
    }
};

BOOST_AUTO_TEST_CASE(mbh_threading_test)
{
    BOOST_CHECK((algorithm{mbh{ts1{}, 5u, 1e-2, 23u}}.get_thread_safety() == thread_safety::basic));
    BOOST_CHECK((algorithm{mbh{ts2{}, 5u, 1e-2, 23u}}.get_thread_safety() == thread_safety::none));
    BOOST_CHECK((algorithm{mbh{ts3{}, 5u, 1e-2, 23u}}.get_thread_safety() == thread_safety::basic));
}

struct ia1 {
    population evolve(const population &pop) const
    {
        return pop;
    }
    double m_data = 0.;
};

BOOST_AUTO_TEST_CASE(mbh_inner_algo_get_test)
{
    // We check that the correct overload is called according to (*this) being const or not
    {
        const mbh uda(ia1{}, 5u, 1e-2, 23u);
        BOOST_CHECK(std::is_const<decltype(uda)>::value);
        BOOST_CHECK(std::is_const<std::remove_reference<decltype(uda.get_inner_algorithm())>::type>::value);
    }
    {
        mbh uda(ia1{}, 5u, 1e-2, 23u);
        BOOST_CHECK(!std::is_const<decltype(uda)>::value);
        BOOST_CHECK(!std::is_const<std::remove_reference<decltype(uda.get_inner_algorithm())>::type>::value);
    }
}
