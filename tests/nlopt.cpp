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

// Disable the checked iterators feature in MSVC. We want it for the source code
// (so it should not be disabled in the headers), but dealing with it in the tests is
// not as useful and quite painful.
#define _SCL_SECURE_NO_WARNINGS

#endif

#define BOOST_TEST_MODULE nlopt_test
#include <boost/test/included/unit_test.hpp>

#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <nlopt.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/luksan_vlcek1.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/serialization.hpp>

using namespace pagmo;

using hs71 = hock_schittkowsky_71;

BOOST_AUTO_TEST_CASE(nlopt_construction)
{
    random_device::set_seed(42);

    algorithm a{nlopt{}};
    BOOST_CHECK_EQUAL(a.extract<nlopt>()->get_solver_name(), "cobyla");
    // Check params of default-constructed instance.
    BOOST_CHECK_EQUAL(boost::any_cast<std::string>(a.extract<nlopt>()->get_selection()), "best");
    BOOST_CHECK_EQUAL(boost::any_cast<std::string>(a.extract<nlopt>()->get_replacement()), "best");
    BOOST_CHECK(a.extract<nlopt>()->get_name() != "");
    BOOST_CHECK(a.extract<nlopt>()->get_extra_info() != "");
    BOOST_CHECK(a.extract<nlopt>()->get_last_opt_result() == NLOPT_SUCCESS);
    BOOST_CHECK_EQUAL(a.extract<nlopt>()->get_stopval(), -HUGE_VAL);
    BOOST_CHECK_EQUAL(a.extract<nlopt>()->get_ftol_rel(), 0.);
    BOOST_CHECK_EQUAL(a.extract<nlopt>()->get_ftol_abs(), 0.);
    BOOST_CHECK_EQUAL(a.extract<nlopt>()->get_xtol_rel(), 1E-8);
    BOOST_CHECK_EQUAL(a.extract<nlopt>()->get_xtol_abs(), 0.);
    BOOST_CHECK_EQUAL(a.extract<nlopt>()->get_maxeval(), 0);
    BOOST_CHECK_EQUAL(a.extract<nlopt>()->get_maxtime(), 0);
    // Change a few params and copy.
    a.extract<nlopt>()->set_selection(12u);
    a.extract<nlopt>()->set_replacement("random");
    a.extract<nlopt>()->set_ftol_abs(1E-5);
    a.extract<nlopt>()->set_maxeval(123);
    // Copy.
    auto b(a);
    BOOST_CHECK_EQUAL(boost::any_cast<population::size_type>(b.extract<nlopt>()->get_selection()), 12u);
    BOOST_CHECK_EQUAL(boost::any_cast<std::string>(b.extract<nlopt>()->get_replacement()), "random");
    BOOST_CHECK(b.extract<nlopt>()->get_last_opt_result() == NLOPT_SUCCESS);
    BOOST_CHECK_EQUAL(b.extract<nlopt>()->get_stopval(), -HUGE_VAL);
    BOOST_CHECK_EQUAL(b.extract<nlopt>()->get_ftol_rel(), 0.);
    BOOST_CHECK_EQUAL(b.extract<nlopt>()->get_ftol_abs(), 1E-5);
    BOOST_CHECK_EQUAL(b.extract<nlopt>()->get_xtol_rel(), 1E-8);
    BOOST_CHECK_EQUAL(b.extract<nlopt>()->get_xtol_abs(), 0.);
    BOOST_CHECK_EQUAL(b.extract<nlopt>()->get_maxeval(), 123);
    BOOST_CHECK_EQUAL(b.extract<nlopt>()->get_maxtime(), 0);
    algorithm c;
    c = b;
    BOOST_CHECK_EQUAL(boost::any_cast<population::size_type>(c.extract<nlopt>()->get_selection()), 12u);
    BOOST_CHECK_EQUAL(boost::any_cast<std::string>(c.extract<nlopt>()->get_replacement()), "random");
    BOOST_CHECK(c.extract<nlopt>()->get_last_opt_result() == NLOPT_SUCCESS);
    BOOST_CHECK_EQUAL(c.extract<nlopt>()->get_stopval(), -HUGE_VAL);
    BOOST_CHECK_EQUAL(c.extract<nlopt>()->get_ftol_rel(), 0.);
    BOOST_CHECK_EQUAL(c.extract<nlopt>()->get_ftol_abs(), 1E-5);
    BOOST_CHECK_EQUAL(c.extract<nlopt>()->get_xtol_rel(), 1E-8);
    BOOST_CHECK_EQUAL(c.extract<nlopt>()->get_xtol_abs(), 0.);
    BOOST_CHECK_EQUAL(c.extract<nlopt>()->get_maxeval(), 123);
    BOOST_CHECK_EQUAL(c.extract<nlopt>()->get_maxtime(), 0);
    // Move.
    auto tmp(*a.extract<nlopt>());
    auto d(std::move(tmp));
    BOOST_CHECK_EQUAL(boost::any_cast<population::size_type>(d.get_selection()), 12u);
    BOOST_CHECK_EQUAL(boost::any_cast<std::string>(d.get_replacement()), "random");
    BOOST_CHECK(d.get_last_opt_result() == NLOPT_SUCCESS);
    BOOST_CHECK_EQUAL(d.get_stopval(), -HUGE_VAL);
    BOOST_CHECK_EQUAL(d.get_ftol_rel(), 0.);
    BOOST_CHECK_EQUAL(d.get_ftol_abs(), 1E-5);
    BOOST_CHECK_EQUAL(d.get_xtol_rel(), 1E-8);
    BOOST_CHECK_EQUAL(d.get_xtol_abs(), 0.);
    BOOST_CHECK_EQUAL(d.get_maxeval(), 123);
    BOOST_CHECK_EQUAL(d.get_maxtime(), 0);
    nlopt e;
    e = std::move(d);
    BOOST_CHECK_EQUAL(boost::any_cast<population::size_type>(e.get_selection()), 12u);
    BOOST_CHECK_EQUAL(boost::any_cast<std::string>(e.get_replacement()), "random");
    BOOST_CHECK(e.get_last_opt_result() == NLOPT_SUCCESS);
    BOOST_CHECK_EQUAL(e.get_stopval(), -HUGE_VAL);
    BOOST_CHECK_EQUAL(e.get_ftol_rel(), 0.);
    BOOST_CHECK_EQUAL(e.get_ftol_abs(), 1E-5);
    BOOST_CHECK_EQUAL(e.get_xtol_rel(), 1E-8);
    BOOST_CHECK_EQUAL(e.get_xtol_abs(), 0.);
    BOOST_CHECK_EQUAL(e.get_maxeval(), 123);
    BOOST_CHECK_EQUAL(e.get_maxtime(), 0);
    // Revive moved-from.
    d = std::move(e);
    BOOST_CHECK_EQUAL(boost::any_cast<population::size_type>(d.get_selection()), 12u);
    BOOST_CHECK_EQUAL(boost::any_cast<std::string>(d.get_replacement()), "random");
    BOOST_CHECK(d.get_last_opt_result() == NLOPT_SUCCESS);
    BOOST_CHECK_EQUAL(d.get_stopval(), -HUGE_VAL);
    BOOST_CHECK_EQUAL(d.get_ftol_rel(), 0.);
    BOOST_CHECK_EQUAL(d.get_ftol_abs(), 1E-5);
    BOOST_CHECK_EQUAL(d.get_xtol_rel(), 1E-8);
    BOOST_CHECK_EQUAL(d.get_xtol_abs(), 0.);
    BOOST_CHECK_EQUAL(d.get_maxeval(), 123);
    BOOST_CHECK_EQUAL(d.get_maxtime(), 0);
    // Check exception throwing on ctor.
    BOOST_CHECK_THROW(nlopt{""}, std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nlopt_selection_replacement)
{
    nlopt a;
    a.set_selection("worst");
    BOOST_CHECK_EQUAL(boost::any_cast<std::string>(a.get_selection()), "worst");
    BOOST_CHECK_THROW(a.set_selection("worstee"), std::invalid_argument);
    a.set_selection(0);
    BOOST_CHECK_EQUAL(boost::any_cast<population::size_type>(a.get_selection()), 0u);
    a.set_replacement("worst");
    BOOST_CHECK_EQUAL(boost::any_cast<std::string>(a.get_replacement()), "worst");
    BOOST_CHECK_THROW(a.set_replacement("worstee"), std::invalid_argument);
    a.set_replacement(0);
    BOOST_CHECK_EQUAL(boost::any_cast<population::size_type>(a.get_replacement()), 0u);
    a.set_random_sr_seed(123);
}

// A version of hs71 which provides the sparsity pattern.
struct hs71a : hs71 {
    sparsity_pattern gradient_sparsity() const
    {
        return detail::dense_gradient(3, 4);
    }
};

BOOST_AUTO_TEST_CASE(nlopt_evolve)
{
    algorithm a{nlopt{"lbfgs"}};
    population pop(rosenbrock{10}, 20);
    a.evolve(pop);
    BOOST_CHECK(a.extract<nlopt>()->get_last_opt_result() >= 0);
    pop = population{zdt{}, 20};
    // MOO not supported by NLopt.
    BOOST_CHECK_THROW(a.evolve(pop), std::invalid_argument);
    // Solver wants gradient, but problem does not provide it.
    pop = population{null_problem{}, 20};
    BOOST_CHECK_THROW(a.evolve(pop), std::invalid_argument);
    pop = population{hs71{}, 20};
    // lbfgs does not support ineq constraints.
    BOOST_CHECK_THROW(a.evolve(pop), std::invalid_argument);
    // mma supports ineq constraints but not eq constraints.
    BOOST_CHECK_THROW(algorithm{nlopt{"mma"}}.evolve(pop), std::invalid_argument);
    a = algorithm{nlopt{"slsqp"}};
    a.extract<nlopt>()->set_verbosity(5);
    for (auto s : {"best", "worst", "random"}) {
        for (auto r : {"best", "worst", "random"}) {
            a.extract<nlopt>()->set_selection(s);
            a.extract<nlopt>()->set_replacement(r);
            pop = population(rosenbrock{10}, 20);
            a.evolve(pop);
            pop = population{hs71{}, 20};
            pop.get_problem().set_c_tol({1E-6, 1E-6});
            a.evolve(pop);
            pop = population{hs71a{}, 20};
            pop.get_problem().set_c_tol({1E-6, 1E-6});
            a.evolve(pop);
        }
    }
    BOOST_CHECK(!a.extract<nlopt>()->get_log().empty());
    for (auto s : {0u, 2u, 15u, 25u}) {
        for (auto r : {1u, 3u, 16u, 25u}) {
            a.extract<nlopt>()->set_selection(s);
            a.extract<nlopt>()->set_replacement(r);
            pop = population(rosenbrock{10}, 20);
            if (s >= 20u || r >= 20u) {
                BOOST_CHECK_THROW(a.evolve(pop), std::invalid_argument);
                continue;
            }
            a.evolve(pop);
            pop = population{hs71{}, 20};
            pop.get_problem().set_c_tol({1E-6, 1E-6});
            a.evolve(pop);
            pop = population{hs71a{}, 20};
            pop.get_problem().set_c_tol({1E-6, 1E-6});
            a.evolve(pop);
        }
    }
    // Empty evolve.
    a.evolve(population{});
    // Invalid initial guesses.
    a = algorithm{nlopt{"slsqp"}};
    pop = population{hs71{}, 1};
    pop.set_x(0, {-123., -123., -123., -123.});
    BOOST_CHECK_THROW(a.evolve(pop), std::invalid_argument);
    pop.set_x(0, {123., 123., 123., 123.});
    BOOST_CHECK_THROW(a.evolve(pop), std::invalid_argument);
    if (std::numeric_limits<double>::has_quiet_NaN) {
        pop.set_x(0, {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()});
        BOOST_CHECK_THROW(a.evolve(pop), std::invalid_argument);
    }
}

BOOST_AUTO_TEST_CASE(nlopt_set_sc)
{
    auto a = nlopt{"slsqp"};
    a.set_stopval(-1.23);
    BOOST_CHECK_EQUAL(a.get_stopval(), -1.23);
    if (std::numeric_limits<double>::has_quiet_NaN) {
        BOOST_CHECK_THROW(a.set_stopval(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    }
    a.set_ftol_rel(-1.23);
    BOOST_CHECK_EQUAL(a.get_ftol_rel(), -1.23);
    if (std::numeric_limits<double>::has_quiet_NaN) {
        BOOST_CHECK_THROW(a.set_ftol_rel(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    }
    a.set_ftol_abs(-1.23);
    BOOST_CHECK_EQUAL(a.get_ftol_abs(), -1.23);
    if (std::numeric_limits<double>::has_quiet_NaN) {
        BOOST_CHECK_THROW(a.set_ftol_abs(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    }
    a.set_xtol_rel(-1.23);
    BOOST_CHECK_EQUAL(a.get_xtol_rel(), -1.23);
    if (std::numeric_limits<double>::has_quiet_NaN) {
        BOOST_CHECK_THROW(a.set_xtol_rel(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    }
    a.set_xtol_abs(-1.23);
    BOOST_CHECK_EQUAL(a.get_xtol_abs(), -1.23);
    if (std::numeric_limits<double>::has_quiet_NaN) {
        BOOST_CHECK_THROW(a.set_xtol_abs(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    }
    a.set_maxtime(123);
}

BOOST_AUTO_TEST_CASE(nlopt_serialization)
{
    for (auto r : {"best", "worst", "random"}) {
        for (auto s : {"best", "worst", "random"}) {
            auto n = nlopt{"slsqp"};
            n.set_replacement(r);
            n.set_selection(s);
            algorithm algo{n};
            algo.set_verbosity(5);
            auto pop = population(hs71{}, 10);
            algo.evolve(pop);
            auto s_log = algo.extract<nlopt>()->get_log();
            // Store the string representation of p.
            std::stringstream ss;
            auto before_text = boost::lexical_cast<std::string>(algo);
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
            BOOST_CHECK_EQUAL(before_text, after_text);
            BOOST_CHECK(s_log == algo.extract<nlopt>()->get_log());
        }
    }
    for (auto r : {0u, 4u, 7u}) {
        for (auto s : {0u, 4u, 7u}) {
            auto n = nlopt{"slsqp"};
            n.set_replacement(r);
            n.set_selection(s);
            algorithm algo{n};
            algo.set_verbosity(5);
            auto pop = population(hs71{}, 10);
            algo.evolve(pop);
            auto s_log = algo.extract<nlopt>()->get_log();
            // Store the string representation of p.
            std::stringstream ss;
            auto before_text = boost::lexical_cast<std::string>(algo);
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
            BOOST_CHECK_EQUAL(before_text, after_text);
            BOOST_CHECK(s_log == algo.extract<nlopt>()->get_log());
        }
    }
}

BOOST_AUTO_TEST_CASE(nlopt_loc_opt)
{
    for (const auto &str : {"auglag", "auglag_eq"}) {
        nlopt n{str};
        n.set_local_optimizer(nlopt{"slsqp"});
        BOOST_CHECK(n.get_local_optimizer());
        BOOST_CHECK(static_cast<const nlopt &>(n).get_local_optimizer());
        // Test serialization.
        algorithm algo{n};
        std::stringstream ss;
        auto before_text = boost::lexical_cast<std::string>(algo);
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
        BOOST_CHECK_EQUAL(before_text, after_text);
        // Test small evolution.
        auto pop = population{hs71{}, 1};
        pop.set_x(0, {2., 2., 2., 2.});
        pop.get_problem().set_c_tol({1E-6, 1E-6});
        algo.evolve(pop);
        BOOST_CHECK(algo.extract<nlopt>()->get_last_opt_result() >= 0);
        // Unset the local optimizer.
        algo.extract<nlopt>()->unset_local_optimizer();
        BOOST_CHECK(!algo.extract<nlopt>()->get_local_optimizer());
        algo.evolve(pop);
        BOOST_CHECK(algo.extract<nlopt>()->get_last_opt_result() == NLOPT_INVALID_ARGS);
        // Auglag inside auglag. Not sure if this is supposed to work, it gives an error
        // currently.
        algo.extract<nlopt>()->set_local_optimizer(nlopt{str});
        algo.extract<nlopt>()->get_local_optimizer()->set_local_optimizer(nlopt{"lbfgs"});
        algo.evolve(pop);
        BOOST_CHECK(algo.extract<nlopt>()->get_last_opt_result() < 0);
    }
    // Check setting a local opt does not do anythig for normal solvers.
    nlopt n{"slsqp"};
    n.set_local_optimizer(nlopt{"lbfgs"});
    algorithm algo{n};
    auto pop = population{rosenbrock{20}, 1};
    algo.evolve(pop);
    BOOST_CHECK(algo.extract<nlopt>()->get_last_opt_result() >= 0);
}
