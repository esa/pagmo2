/* Copyright 2017-2026 PaGMO development team

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

#define BOOST_TEST_MODULE sga_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/lexical_cast.hpp>

#include <chrono>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sea.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowski_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/minlp_rastrigin.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(sga_algorithm_construction)
{
    // sga uda{1u, 0.5, 10., .02, .5, 1u, 2u, "gaussian", "truncated", "sbx", 0u};
    // We check the default constructor, a correct call and the possibility to build a pagmo::algorithm
    BOOST_CHECK_NO_THROW(sga{});
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "gaussian", "tournament", 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "uniform", "tournament", 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, 20., 5u, "exponential", "polynomial", "tournament", 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "gaussian", "truncated", 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 5u, "binomial", "gaussian", "tournament", 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 5u, "sbx", "gaussian", "tournament", 32u}));
    BOOST_CHECK_NO_THROW(algorithm(sga{}));
    // We check incorrect calls to the constructor
    BOOST_CHECK_THROW((sga{1u, 12., 10., .02, .5, 5u, "exponential", "gaussian", "tournament", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, -1.1, 10., .02, .5, 5u, "exponential", "gaussian", "tournament", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 0.1, .02, .5, 5u, "exponential", "gaussian", "truncated", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 101., .02, .5, 5u, "exponential", "gaussian", "tournament", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., -0.2, .5, 5u, "exponential", "gaussian", "truncated", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., 1.3, .5, 5u, "exponential", "gaussian", "tournament", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "unknown_method", "tournament", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "gaussian", "unknown_method", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 5u, "unknown_method", "gaussian", "truncated", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "polynomial", "tournament", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, 101, 5u, "exponential", "polynomial", "truncated", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, -3, 5u, "exponential", "uniform", "tournament", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, std::numeric_limits<double>::quiet_NaN(), 10., .02, .5, 5u, "exponential", "gaussian",
                           "tournament", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, 1.1, 5u, "exponential", "uniform", "tournament", 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, 0.9, 0u, "exponential", "uniform", "tournament", 32u}),
                      std::invalid_argument);
}
BOOST_AUTO_TEST_CASE(sga_evolve_test)
{
    // We start testing the various throws in case the evolve is called on a population
    // not suitable for sga
    BOOST_CHECK_THROW((sga{}.evolve(population{zdt{}, 5u, 23u})), std::invalid_argument);
    BOOST_CHECK_THROW((sga{}.evolve(population{hock_schittkowski_71{}, 5u, 23u})), std::invalid_argument);
    BOOST_CHECK_THROW((sga{}.evolve(population{schwefel{20u}, 1u, 23u})), std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 21u, "exponential", "gaussian", "tournament", 32u}.evolve(
                          population{schwefel{20u}, 20u, 23u})),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 2u, "sbx", "gaussian", "tournament", 32u}.evolve(
                          population{schwefel{20u}, 25u, 23u})),
                      std::invalid_argument);
    // And a clean exit for 0 generations
    population pop{schwefel{25u}, 10u};
    BOOST_CHECK(sga{0u}.evolve(pop).get_x()[0] == pop.get_x()[0]);
    // The we test that evolution is deterministic if the
    // seed is controlled.
    std::vector<sga> udas = {
        sga{10u, .90, 1., 0.1, 1., 2u, "exponential", "gaussian", "tournament", 32u},
        sga{10u, .90, 1., 0.1, 1., 2u, "sbx", "gaussian", "tournament", 32u},
        sga{10u, .90, 1., 0.1, 1., 2u, "binomial", "gaussian", "tournament", 32u},
        sga{10u, .90, 1., 0.1, 1., 2u, "single", "gaussian", "tournament", 32u},
        sga{10u, .90, 1., 0.1, 1., 2u, "exponential", "polynomial", "tournament", 32u},
        sga{10u, .90, 1., 0.1, 1., 2u, "exponential", "uniform", "tournament", 32u},
        sga{10u, .90, 1., 0.1, 1., 2u, "exponential", "gaussian", "truncated", 32u},
    };
    // On a deterministic problem
    for (sga &uda : udas) {
        problem prob{schwefel{20u}};
        population pop1{prob, 20u, 23u};
        uda.set_seed(32u);
        uda.set_verbosity(1u);
        pop1 = uda.evolve(pop1);
        auto log1 = uda.get_log();

        population pop2{prob, 20u, 23u};
        uda.set_seed(32u);
        pop2 = uda.evolve(pop2);
        auto log2 = uda.get_log();

        BOOST_CHECK(log1 == log2);
    }
    // And on a stochastic one
    for (sga &uda : udas) {
        problem prob{inventory{25u, 5u, 1432u}};
        population pop1{prob, 20u, 23u};
        uda.set_seed(32u);
        uda.set_verbosity(1u);
        pop1 = uda.evolve(pop1);
        auto log1 = uda.get_log();

        population pop2{prob, 20u, 23u};
        uda.set_seed(32u);
        pop2 = uda.evolve(pop2);
        auto log2 = uda.get_log();

        BOOST_CHECK(log1 == log2);
    }
    // And on a MINLP
    for (sga &uda : udas) {
        problem prob{minlp_rastrigin{5u, 5u}};
        population pop1{prob, 20u, 23u};
        uda.set_seed(32u);
        uda.set_verbosity(1u);
        pop1 = uda.evolve(pop1);
        auto log1 = uda.get_log();

        population pop2{prob, 20u, 23u};
        uda.set_seed(32u);
        pop2 = uda.evolve(pop2);
        auto log2 = uda.get_log();

        BOOST_CHECK(log1 == log2);
    }
    // We call the extra info and chech they do not throw
    for (sga &uda : udas) {
        BOOST_CHECK_NO_THROW(uda.get_extra_info());
    }
}

BOOST_AUTO_TEST_CASE(sga_serialization_test)
{
    // Make one evolution
    problem prob{schwefel{20u}};
    population pop{prob, 20u, 23u};
    algorithm algo{sga{10u}};
    algo.extract<sga>()->set_max_time(1e6);
    algo.set_seed(32u);

    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<sga>()->get_log();
    // Now serialize, deserialize and compare the result.
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << algo;
    }
    // Change the content of p before deserializing.
    algo = algorithm{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> algo;
    }
    auto after_text = boost::lexical_cast<std::string>(algo);
    auto after_log = algo.extract<sga>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    BOOST_CHECK(before_log == after_log);
    // so we implement a close check

    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_EQUAL(std::get<1>(before_log[i]), std::get<1>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<2>(before_log[i]), std::get<2>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<3>(before_log[i]), std::get<3>(after_log[i]), 1e-8);
    }
}

BOOST_AUTO_TEST_CASE(sga_out_of_bounds_population_test)
{
    // Out-of-bounds/non-finite initial individuals must be rejected (see issue #567):
    // the crossover and mutation operators assume feasible parents.
    problem prob{rosenbrock{}};
    population pop{prob, 10u, 23u};
    auto x = pop.get_x()[0];
    auto f = pop.get_f()[0];
    x[0] = -6.0; // below the lower bound (-5)
    pop.set_xf(0, x, f);
    BOOST_CHECK_THROW(sga{1u}.evolve(pop), std::invalid_argument);
    x[0] = 11.0; // above the upper bound (10)
    pop.set_xf(0, x, f);
    BOOST_CHECK_THROW(sga{1u}.evolve(pop), std::invalid_argument);
    x[0] = std::numeric_limits<double>::quiet_NaN();
    pop.set_xf(0, x, f);
    BOOST_CHECK_THROW(sga{1u}.evolve(pop), std::invalid_argument);
    x[0] = std::numeric_limits<double>::infinity();
    pop.set_xf(0, x, f);
    BOOST_CHECK_THROW(sga{1u}.evolve(pop), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(sga_time_limit_test)
{
    // The time limit defaults to zero (no limit) and must be a non-negative finite number of seconds.
    BOOST_CHECK_EQUAL(sga{}.get_max_time(), 0.);
    BOOST_CHECK_EQUAL((sga{10u, .90, 1., 0.02, 1., 2u, "exponential", "polynomial", "tournament", 23u, 1.5}).get_max_time(), 1.5);
    for (auto max_time : {-1., std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        BOOST_CHECK_THROW((sga{10u, .90, 1., 0.02, 1., 2u, "exponential", "polynomial", "tournament", 23u, max_time}), std::invalid_argument);
    }
    sga uda{};
    uda.set_max_time(2.5);
    BOOST_CHECK_EQUAL(uda.get_max_time(), 2.5);
    BOOST_CHECK_THROW(uda.set_max_time(-1.), std::invalid_argument);
    BOOST_CHECK_THROW(uda.set_max_time(std::numeric_limits<double>::infinity()), std::invalid_argument);
    BOOST_CHECK_EQUAL(uda.get_max_time(), 2.5);
    BOOST_CHECK(uda.get_extra_info().find("Max time") != std::string::npos);

    population pop{rosenbrock{10u}, 20u, 23u};
    // A time limit that is not reached leaves the evolution unchanged.
    {
        auto uda1 = sga{20u, .90, 1., 0.02, 1., 2u, "exponential", "polynomial", "tournament", 23u, 0.};
        auto uda2 = sga{20u, .90, 1., 0.02, 1., 2u, "exponential", "polynomial", "tournament", 23u, 1e6};
        uda1.set_verbosity(1u);
        uda2.set_verbosity(1u);
        const auto pop1 = uda1.evolve(pop);
        const auto pop2 = uda2.evolve(pop);
        BOOST_CHECK(pop1.get_x() == pop2.get_x());
        BOOST_CHECK(pop1.get_f() == pop2.get_f());
        BOOST_CHECK(uda1.get_log() == uda2.get_log());
    }
    // The time limit stops the evolution long before the generation limit, and each call to evolve() has
    // its own time budget.
    {
        const double max_time = 0.1;
        auto uda1 = sga{std::numeric_limits<unsigned>::max(), .90, 1., 0.02, 1., 2u, "exponential", "polynomial", "tournament", 23u, max_time};
        auto pop1 = pop;
        for (auto i = 0; i < 2; ++i) {
            const auto start = std::chrono::steady_clock::now();
            pop1 = uda1.evolve(pop1);
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            BOOST_CHECK_GE(elapsed, max_time);
            BOOST_CHECK_LT(elapsed, max_time + 5.);
        }
    }
    // However small the time limit, the first generation is always run.
    {
        auto uda1 = sga{std::numeric_limits<unsigned>::max(), .90, 1., 0.02, 1., 2u, "exponential", "polynomial", "tournament", 23u, 1e-9};
        auto uda2 = sga{1u, .90, 1., 0.02, 1., 2u, "exponential", "polynomial", "tournament", 23u, 0.};
        const auto pop1 = uda1.evolve(pop);
        const auto pop2 = uda2.evolve(pop);
        BOOST_CHECK(pop1.get_x() == pop2.get_x());
        BOOST_CHECK(pop1.get_f() == pop2.get_f());
    }
    // When verbose, the time limit is reported as the only exit condition.
    {
        auto uda1 = sga{std::numeric_limits<unsigned>::max(),
                        .90,
                        1.,
                        0.02,
                        1.,
                        2u,
                        "exponential",
                        "polynomial",
                        "tournament",
                        23u,
                        1e-9};
        uda1.set_verbosity(1u);
        std::ostringstream output;
        auto *const cout_buf = std::cout.rdbuf(output.rdbuf());
        uda1.evolve(pop);
        std::cout.rdbuf(cout_buf);
        const auto out = output.str();
        const auto pos = out.find("Exit condition -- max_time = ");
        BOOST_CHECK(pos != std::string::npos);
        BOOST_CHECK(out.find("Exit condition") == pos);
        BOOST_CHECK(out.find("Exit condition", pos + 1u) == std::string::npos);
    }
}
