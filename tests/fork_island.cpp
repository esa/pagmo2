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

#define BOOST_TEST_MODULE fork_island_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <chrono>
#include <csignal>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <sys/types.h>

#include <boost/algorithm/string/predicate.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/island.hpp>
#include <pagmo/islands/fork_island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

// A silly problem that, after max fitness evals, just waits.
struct godot1 {
    explicit godot1(unsigned max) : m_max(max), m_counter(0) {}
    godot1() : godot1(0) {}
    vector_double fitness(const vector_double &) const
    {
        if (m_max == m_counter++) {
            std::this_thread::sleep_for(std::chrono::seconds(3600));
        }
        return {.5};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        detail::archive(ar, m_max, m_counter);
    }
    unsigned m_max;
    mutable unsigned m_counter;
};

PAGMO_S11N_PROBLEM_EXPORT(godot1)

BOOST_AUTO_TEST_CASE(fork_island_basic)
{
    {
        fork_island fi_0;
        BOOST_CHECK(fi_0.get_child_pid() == pid_t(0));
        fork_island fi_1(fi_0), fi_2(std::move(fi_0));
        BOOST_CHECK(fi_1.get_child_pid() == pid_t(0));
        BOOST_CHECK(fi_2.get_child_pid() == pid_t(0));
        BOOST_CHECK(boost::contains(fi_0.get_extra_info(), "No active child"));
        BOOST_CHECK(boost::contains(fi_1.get_extra_info(), "No active child"));
        BOOST_CHECK(boost::contains(fi_2.get_extra_info(), "No active child"));
        BOOST_CHECK_EQUAL(fi_0.get_name(), "Fork island");
    }
    // NOTE: on recent OSX versions, the fork() behaviour changed and trying
    // to do error handling via exceptions in the forked() process now does
    // not seem to work any more. Let's disable the error handling tests
    // for now, perhaps we can investigate this further in the future.
#if !defined(__APPLE__)
    {
        // Test: try to kill a running island.
        island fi_0(fork_island{}, de{200}, godot1{20}, 20);
        BOOST_CHECK(fi_0.extract<fork_island>() != nullptr);
        BOOST_CHECK(boost::contains(fi_0.get_extra_info(), "No active child"));
        fi_0.evolve();
        // Busy wait until the child is running.
        pid_t child_pid;
        while (!(child_pid = fi_0.extract<fork_island>()->get_child_pid())) {
        }
        BOOST_CHECK(boost::contains(fi_0.get_extra_info(), "Child PID:"));
        // """
        // Kill the boy and let the man be born.
        // """
        kill(child_pid, SIGTERM);
        // Check that killing the child raised an error in the parent process.
        BOOST_CHECK_THROW(fi_0.wait_check(), std::exception);
        BOOST_CHECK(boost::contains(fi_0.get_extra_info(), "No active child"));
    }
    {
        // Test: try to generate an error in the evolution.
        // NOTE: de wants more than 1 individual in the pop.
        island fi_0(fork_island{}, de{1}, rosenbrock{}, 1);
        BOOST_CHECK(fi_0.extract<fork_island>() != nullptr);
        BOOST_CHECK(boost::contains(fi_0.get_extra_info(), "No active child"));
        fi_0.evolve();
        BOOST_CHECK_EXCEPTION(fi_0.wait_check(), std::runtime_error, [](const std::runtime_error &re) {
            return boost::contains(re.what(), "needs at least 5 individuals in the population");
        });
    }
#endif
}

// Check that the population actually evolves.
BOOST_AUTO_TEST_CASE(fork_island_evolve)
{
    island fi_0(fork_island{}, compass_search{100}, rosenbrock{}, 1, 0);
    const auto old_cf = fi_0.get_population().champion_f();
    fi_0.evolve();
    fi_0.wait_check();
    const auto new_cf = fi_0.get_population().champion_f();
    BOOST_CHECK(new_cf[0] < old_cf[0]);
}

// An algorithm that changes its state at every evolve() call.
struct stateful_algo {
    population evolve(const population &pop) const
    {
        ++n_evolve;
        return pop;
    }
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &n_evolve;
    }
    mutable int n_evolve = 0;
};

PAGMO_S11N_ALGORITHM_EXPORT(stateful_algo)

// Check that the state of the algorithm is preserved.
BOOST_AUTO_TEST_CASE(fork_island_stateful_algo)
{
    island fi_0(fork_island{}, stateful_algo{}, rosenbrock{}, 1, 0);
    BOOST_CHECK(fi_0.get_algorithm().extract<stateful_algo>()->n_evolve == 0);
    fi_0.evolve();
    fi_0.wait_check();
    BOOST_CHECK(fi_0.get_algorithm().extract<stateful_algo>()->n_evolve == 1);
}

struct recursive_algo1 {
    population evolve(const population &pop) const
    {
        island fi_0(fork_island{}, compass_search{100}, pop);
        fi_0.evolve();
        fi_0.wait_check();
        return fi_0.get_population();
    }
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

struct recursive_algo2 {
    population evolve(const population &pop) const
    {
        island fi_0(fork_island{}, de{1}, pop);
        fi_0.evolve();
        fi_0.wait_check();
        return fi_0.get_population();
    }
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

PAGMO_S11N_ALGORITHM_EXPORT(recursive_algo1)
PAGMO_S11N_ALGORITHM_EXPORT(recursive_algo2)

// Try to call fork() inside fork().
BOOST_AUTO_TEST_CASE(fork_island_recurse)
{
    {
        island fi_0(fork_island{}, recursive_algo1{}, rosenbrock{}, 1, 0);
        const auto old_cf = fi_0.get_population().champion_f();
        fi_0.evolve();
        fi_0.wait_check();
        const auto new_cf = fi_0.get_population().champion_f();
        BOOST_CHECK(new_cf[0] < old_cf[0]);
    }
#if !defined(__APPLE__)
    {
        // Try also error transport.
        island fi_0(fork_island{}, recursive_algo2{}, rosenbrock{}, 1, 0);
        fi_0.evolve();
        BOOST_CHECK_EXCEPTION(fi_0.wait_check(), std::runtime_error, [](const std::runtime_error &re) {
            return boost::contains(re.what(), "needs at least 5 individuals in the population");
        });
    }
#endif
}

// Run a moderate amount of fork islands in parallel.
BOOST_AUTO_TEST_CASE(fork_island_torture)
{
    std::vector<island> visl(100u, island(fork_island{}, compass_search{100}, rosenbrock{100}, 50, 0));
    for (auto &isl : visl) {
        isl.evolve();
    }
    for (auto &isl : visl) {
        BOOST_CHECK_NO_THROW(isl.wait_check());
    }
}
