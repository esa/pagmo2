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

#define BOOST_TEST_MODULE ipopt_test
#include <boost/test/included/unit_test.hpp>

#include <IpReturnCodes.hpp>
#include <boost/lexical_cast.hpp>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/ipopt.hpp>
#include <pagmo/config.hpp>
#include <pagmo/island.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/luksan_vlcek1.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#if defined(PAGMO_WITH_FORK_ISLAND)
#include <pagmo/islands/fork_island.hpp>
#else
#include <pagmo/islands/thread_island.hpp>
#endif

using namespace pagmo;

// NOTE: this test checks that the output of the various virtual methods implemented
// from the C++ Ipopt API is consistent with the manually-coded hs71 problem from the official
// Ipopt documentation (bits of which were copy-pasted here).
BOOST_AUTO_TEST_CASE(ipopt_nlp_test)
{
    random_device::set_seed(42);

    using ipopt_nlp = detail::ipopt_nlp;
    using Index = ipopt_nlp::Index;
    problem prob(hock_schittkowsky_71{});
    ipopt_nlp nlp(prob, {1.1, 1.2, 1.3, 1.4}, 5u);
    BOOST_CHECK_EQUAL(nlp.m_verbosity, 5u);

    // Test info getting.
    Index n, m, nnz_jac_g, nnz_h_lag;
    ipopt_nlp::IndexStyleEnum index_style;
    BOOST_CHECK(nlp.get_nlp_info(n, m, nnz_jac_g, nnz_h_lag, index_style));
    BOOST_CHECK_EQUAL(n, Index(4));
    BOOST_CHECK_EQUAL(m, Index(2));
    BOOST_CHECK_EQUAL(nnz_jac_g, Index(8));
    BOOST_CHECK_EQUAL(nnz_h_lag, Index(10));
    BOOST_CHECK(index_style == ipopt_nlp::C_STYLE);

    // Bounds.
    vector_double lb(4), ub(4), c_lb(2), c_ub(2);
    nlp.get_bounds_info(4, lb.data(), ub.data(), 2, c_lb.data(), c_ub.data());
    BOOST_CHECK((lb == vector_double{1., 1., 1., 1.}));
    BOOST_CHECK((ub == vector_double{5., 5., 5., 5.}));
    BOOST_CHECK(
        (c_lb
         == vector_double{0., std::numeric_limits<double>::has_infinity ? -std::numeric_limits<double>::infinity()
                                                                        : std::numeric_limits<double>::lowest()}));
    BOOST_CHECK((c_ub == vector_double{0., 0.}));

    // Initial guess.
    vector_double start(4);
    nlp.get_starting_point(4, true, start.data(), false, nullptr, nullptr, 2, false, nullptr);
    BOOST_CHECK((start == vector_double{1.1, 1.2, 1.3, 1.4}));

    // eval_f().
    double objval;
    const vector_double x{2.1, 2.2, 2.3, 2.4};
    nlp.eval_f(4, x.data(), true, objval);
    BOOST_CHECK_EQUAL(prob.fitness(x)[0], objval);

    // eval_grad_f().
    vector_double grad_f(4);
    nlp.eval_grad_f(4, x.data(), true, grad_f.data());
    auto grad_f_copy(grad_f);
    // Compute manually and compare.
    grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
    BOOST_CHECK(std::abs(grad_f[0] - grad_f_copy[0]) < 1E-8);
    grad_f[1] = x[0] * x[3];
    BOOST_CHECK(std::abs(grad_f[1] - grad_f_copy[1]) < 1E-8);
    grad_f[2] = x[0] * x[3] + 1;
    BOOST_CHECK(std::abs(grad_f[2] - grad_f_copy[2]) < 1E-8);
    grad_f[3] = x[0] * (x[0] + x[1] + x[2]);
    BOOST_CHECK(std::abs(grad_f[3] - grad_f_copy[3]) < 1E-8);

    // eval_g().
    vector_double g(2);
    nlp.eval_g(4, x.data(), true, 2, g.data());
    BOOST_CHECK(std::abs(g[0] - (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.)) < 1E-8);
    BOOST_CHECK(std::abs(g[1] - (25. - x[0] * x[1] * x[2] * x[3])) < 1E-8);

    // eval_jac_g().
    std::vector<Index> iRow(8), jCol(8);
    vector_double jac_g(8);
    // Get the sparsity pattern first.
    nlp.eval_jac_g(4, x.data(), true, 2, 8, iRow.data(), jCol.data(), nullptr);
    BOOST_CHECK((iRow == std::vector<Index>{0, 0, 0, 0, 1, 1, 1, 1}));
    BOOST_CHECK((jCol == std::vector<Index>{0, 1, 2, 3, 0, 1, 2, 3}));
    // Jacobian now.
    nlp.eval_jac_g(4, x.data(), true, 2, 8, iRow.data(), jCol.data(), jac_g.data());
    BOOST_CHECK(std::abs(jac_g[0] - (2 * x[0])) < 1E-8);
    BOOST_CHECK(std::abs(jac_g[1] - (2 * x[1])) < 1E-8);
    BOOST_CHECK(std::abs(jac_g[2] - (2 * x[2])) < 1E-8);
    BOOST_CHECK(std::abs(jac_g[3] - (2 * x[3])) < 1E-8);
    BOOST_CHECK(std::abs(jac_g[4] - (-x[1] * x[2] * x[3])) < 1E-8);
    BOOST_CHECK(std::abs(jac_g[5] - (-x[0] * x[2] * x[3])) < 1E-8);
    BOOST_CHECK(std::abs(jac_g[6] - (-x[0] * x[1] * x[3])) < 1E-8);
    BOOST_CHECK(std::abs(jac_g[7] - (-x[0] * x[1] * x[2])) < 1E-8);

    // eval_h().
    const vector_double lambda{2., 3.};
    vector_double h(10);
    // Get the sparsity pattern first.
    const auto dhess = detail::dense_hessian(4);
    iRow.resize(static_cast<decltype(iRow.size())>(dhess.size()));
    jCol.resize(static_cast<decltype(jCol.size())>(dhess.size()));
    const double obj_factor = 1.5;
    nlp.eval_h(4, x.data(), true, obj_factor, 2, lambda.data(), true, 10, iRow.data(), jCol.data(), nullptr);
    Index idx = 0;
    for (Index row = 0; row < 4; row++) {
        for (Index col = 0; col <= row; col++) {
            BOOST_CHECK(iRow.data()[idx] == row);
            BOOST_CHECK(jCol.data()[idx] == col);
            idx++;
        }
    }
    // The value now.
    nlp.eval_h(4, x.data(), true, 1.5, 2, lambda.data(), true, 10, iRow.data(), jCol.data(), h.data());
    BOOST_CHECK(std::abs(h[0] - (obj_factor * (2 * x[3]) + lambda[0] * 2)) < 1E-8);
    BOOST_CHECK(std::abs(h[1] - (obj_factor * (x[3]) - lambda[1] * (x[2] * x[3]))) < 1E-8);
    BOOST_CHECK(std::abs(h[2] - (0. + lambda[0] * 2)) < 1E-8);
    BOOST_CHECK(std::abs(h[3] - (obj_factor * (x[3]) - lambda[1] * (x[1] * x[3]))) < 1E-8);
    BOOST_CHECK(std::abs(h[4] - (0. - lambda[1] * (x[0] * x[3]))) < 1E-8);
    BOOST_CHECK(std::abs(h[5] - (0. + lambda[0] * 2)) < 1E-8);
    BOOST_CHECK(std::abs(h[6] - (obj_factor * (2 * x[0] + x[1] + x[2]) - lambda[1] * (x[1] * x[2]))) < 1E-8);
    BOOST_CHECK(std::abs(h[7] - (obj_factor * (x[0]) - lambda[1] * (x[0] * x[2]))) < 1E-8);
    BOOST_CHECK(std::abs(h[8] - (obj_factor * (x[0]) - lambda[1] * (x[0] * x[1]))) < 1E-8);
    BOOST_CHECK(std::abs(h[9] - (0. + lambda[0] * 2)) < 1E-8);
}

BOOST_AUTO_TEST_CASE(ipopt_evolve_test_00)
{
    ipopt ip;
    algorithm algo(ip);
    algo.set_verbosity(1);
    problem prob(hock_schittkowsky_71{});
    prob.set_c_tol({1E-8, 1E-8});
    population pop(prob, 1);
    algo.evolve(pop);
}

BOOST_AUTO_TEST_CASE(ipopt_evolve_test_01)
{
    ipopt ip;
    algorithm algo(ip);
    algo.set_verbosity(1);
    problem prob(luksan_vlcek1{4});
    prob.set_c_tol({1E-8, 1E-8});
    population pop(prob, 1);
    algo.evolve(pop);
    BOOST_CHECK_EQUAL(Ipopt::Solve_Succeeded, algo.extract<ipopt>()->get_last_opt_result());
    BOOST_CHECK(!algo.get_extra_info().empty());
    BOOST_CHECK(!algo.extract<ipopt>()->get_log().empty());
}

// Empty pop.
BOOST_AUTO_TEST_CASE(ipopt_evolve_test_02)
{
    ipopt ip;
    algorithm algo(ip);
    algo.set_verbosity(1);
    problem prob(luksan_vlcek1{4});
    prob.set_c_tol({1E-8, 1E-8});
    population pop(prob, 0);
    algo.evolve(pop);
}

struct throw_hs71_0 : hock_schittkowsky_71 {
    vector_double fitness(const vector_double &dv) const
    {
        if (counter == 5u) {
            throw std::invalid_argument("");
        }
        ++counter;
        return static_cast<const hock_schittkowsky_71 *>(this)->fitness(dv);
    }
    mutable unsigned counter = 0;
};

struct throw_hs71_1 : hock_schittkowsky_71 {
    vector_double gradient(const vector_double &dv) const
    {
        if (counter == 5u) {
            throw std::invalid_argument("");
        }
        ++counter;
        return static_cast<const hock_schittkowsky_71 *>(this)->gradient(dv);
    }
    mutable unsigned counter = 0;
};

struct throw_hs71_2 : hock_schittkowsky_71 {
    vector_double gradient(const vector_double &dv) const
    {
        if (counter == 6u) {
            throw std::invalid_argument("");
        }
        ++counter;
        return static_cast<const hock_schittkowsky_71 *>(this)->gradient(dv);
    }
    mutable unsigned counter = 0;
};

BOOST_AUTO_TEST_CASE(ipopt_failure_modes)
{
    {
        // Multiobjective.
        algorithm algo(ipopt{});
        population pop(zdt{}, 1);
        algo.extract<ipopt>()->set_selection("random");
        BOOST_CHECK_THROW(algo.evolve(pop), std::invalid_argument);
    }
    {
        // Problem does not provide gradient.
        algorithm algo(ipopt{});
        population pop(schwefel{20}, 1);
        BOOST_CHECK_THROW(algo.evolve(pop), std::invalid_argument);
    }
    {
        // Problem's objfun throws.
        algorithm algo(ipopt{});
        population pop(throw_hs71_0{}, 1);
        BOOST_CHECK_THROW(algo.evolve(pop), std::invalid_argument);
    }
    {
        // Problem's gradient throws.
        algorithm algo(ipopt{});
        population pop(throw_hs71_1{}, 1);
        BOOST_CHECK_THROW(algo.evolve(pop), std::invalid_argument);
    }
    {
        // Problem's gradient throws.
        algorithm algo(ipopt{});
        population pop(throw_hs71_2{}, 1);
        BOOST_CHECK_THROW(algo.evolve(pop), std::invalid_argument);
    }
    {
        // Explicitly requiring exact hessians in a problem that does
        // not provide them.
        algorithm algo(ipopt{});
        algo.extract<ipopt>()->set_string_option("hessian_approximation", "exact");
        population pop(luksan_vlcek1{}, 1);
        BOOST_CHECK_THROW(algo.evolve(pop), std::invalid_argument);
    }
    {
        // Set bogus string option.
        algorithm algo(ipopt{});
        population pop(hock_schittkowsky_71{}, 1);
        algo.extract<ipopt>()->set_string_option("hello,", "world");
        BOOST_CHECK_THROW(algo.evolve(pop), std::invalid_argument);
    }
    {
        // Set bogus integer option.
        algorithm algo(ipopt{});
        population pop(hock_schittkowsky_71{}, 1);
        algo.extract<ipopt>()->set_integer_option("hello, world", 3);
        BOOST_CHECK_THROW(algo.evolve(pop), std::invalid_argument);
    }
    {
        // Set bogus numeric option.
        algorithm algo(ipopt{});
        population pop(hock_schittkowsky_71{}, 1);
        algo.extract<ipopt>()->set_numeric_option("hello, world", 3.);
        BOOST_CHECK_THROW(algo.evolve(pop), std::invalid_argument);
    }
    {
        // Initial guess out of bounds.
        algorithm algo(ipopt{});
        population pop(hock_schittkowsky_71{}, 1);
        pop.set_x(0, {-100., -100., -100., -100.});
        BOOST_CHECK_THROW(algo.evolve(pop), std::invalid_argument);
    }
    if (std::numeric_limits<double>::has_quiet_NaN) {
        // Initial guess has nans.
        algorithm algo(ipopt{});
        population pop(hock_schittkowsky_71{}, 1);
        pop.set_x(0, {2., 2., 2., std::numeric_limits<double>::quiet_NaN()});
        BOOST_CHECK_THROW(algo.evolve(pop), std::invalid_argument);
    }
}

// A problem that provides hessians but not their sparsity.
struct hs71_no_sp : hock_schittkowsky_71 {
    bool has_hessians_sparsity() const
    {
        return false;
    }
    std::vector<vector_double> hessians(const vector_double &x) const
    {
        return {{2 * x[3], x[3], 0., x[3], 0., 0., 2 * x[0] + x[1] + x[2], x[0], x[0], 0.},
                {2., 0., 2., 0., 0., 2., 0., 0., 0., 2.},
                {0., -x[2] * x[3], 0., -x[1] * x[3], -x[0] * x[3], 0., -x[1] * x[2], -x[0] * x[2], -x[0] * x[1], 0.}};
    }
};

BOOST_AUTO_TEST_CASE(ipopt_hess_not_sp)
{
    ipopt ip;
    algorithm algo(ip);
    algo.set_verbosity(1);
    problem prob(hs71_no_sp{});
    prob.set_c_tol({1E-8, 1E-8});
    population pop(prob, 1);
    algo.evolve(pop);
}

BOOST_AUTO_TEST_CASE(ipopt_serialization)
{
    for (auto r : {"best", "worst", "random"}) {
        for (auto s : {"best", "worst", "random"}) {
            auto n = ipopt{};
            n.set_replacement(r);
            n.set_selection(s);
            algorithm algo{n};
            algo.set_verbosity(5);
            auto pop = population(hock_schittkowsky_71{}, 10);
            algo.evolve(pop);
            auto s_log = algo.extract<ipopt>()->get_log();
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
            BOOST_CHECK(s_log == algo.extract<ipopt>()->get_log());
        }
    }
    for (auto r : {0u, 4u, 7u}) {
        for (auto s : {0u, 4u, 7u}) {
            auto n = ipopt{};
            n.set_replacement(r);
            n.set_selection(s);
            algorithm algo{n};
            algo.set_verbosity(5);
            auto pop = population(hock_schittkowsky_71{}, 10);
            algo.evolve(pop);
            auto s_log = algo.extract<ipopt>()->get_log();
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
            BOOST_CHECK(s_log == algo.extract<ipopt>()->get_log());
        }
    }
}

BOOST_AUTO_TEST_CASE(ipopt_options)
{
    ipopt ip;
    // String.
    ip.set_string_option("bart", "simpson");
    ip.set_string_options({{"homer", "simpson"}, {"marge", "simpson"}});
    BOOST_CHECK(
        (ip.get_string_options()
         == std::map<std::string, std::string>{{"bart", "simpson"}, {"homer", "simpson"}, {"marge", "simpson"}}));
    ip.reset_string_options();
    BOOST_CHECK(ip.get_string_options().empty());
    // Integer.
    ip.set_integer_option("bart", 1);
    ip.set_integer_options({{"homer", 2}, {"marge", 3}});
    BOOST_CHECK(
        (ip.get_integer_options() == std::map<std::string, Ipopt::Index>{{"bart", 1}, {"homer", 2}, {"marge", 3}}));
    ip.reset_integer_options();
    BOOST_CHECK(ip.get_integer_options().empty());
    // Numeric.
    ip.set_numeric_option("bart", 1);
    ip.set_numeric_options({{"homer", 2}, {"marge", 3}});
    BOOST_CHECK((ip.get_numeric_options() == std::map<std::string, double>{{"bart", 1}, {"homer", 2}, {"marge", 3}}));
    ip.reset_numeric_options();
    BOOST_CHECK(ip.get_numeric_options().empty());
}

BOOST_AUTO_TEST_CASE(ipopt_thread_safety)
{
    BOOST_CHECK(algorithm(ipopt{}).get_thread_safety() == thread_safety::none);
    // Check the island selection type.
#if defined(PAGMO_WITH_FORK_ISLAND)
    BOOST_CHECK((island{ipopt{}, luksan_vlcek1{4}, 10}.is<fork_island>()));
#else
    BOOST_CHECK((island{ipopt{}, luksan_vlcek1{4}, 10}.is<thread_island>()));
#endif
}
