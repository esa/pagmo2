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

#define BOOST_TEST_MODULE fork_island_test
#include <boost/test/included/unit_test.hpp>

#include <chrono>
#include <csignal>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>

#include <sys/types.h>

#include <boost/algorithm/string/predicate.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/island.hpp>
#include <pagmo/islands/fork_island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/serialization.hpp>
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
    void serialize(Archive &ar)
    {
        ar(m_max, m_counter);
    }
    unsigned m_max;
    mutable unsigned m_counter;
};

PAGMO_REGISTER_PROBLEM(godot1)

BOOST_AUTO_TEST_CASE(fork_island_basic)
{
    {
        fork_island fi_0;
        BOOST_CHECK(fi_0.get_child_pid() == pid_t(0));
        fork_island fi_1(fi_0), fi_2(std::move(fi_0));
        BOOST_CHECK(fi_1.get_child_pid() == pid_t(0));
        BOOST_CHECK(fi_2.get_child_pid() == pid_t(0));
        BOOST_CHECK(boost::contains(fi_0.get_extra_info(), "No active child."));
        BOOST_CHECK(boost::contains(fi_1.get_extra_info(), "No active child."));
        BOOST_CHECK(boost::contains(fi_2.get_extra_info(), "No active child."));
        BOOST_CHECK_EQUAL(fi_0.get_name(), "Fork island");
    }
    {
        // Test: try to kill a running island.
        island fi_0(fork_island{}, de{200}, godot1{20}, 20);
        BOOST_CHECK(fi_0.extract<fork_island>() != nullptr);
        BOOST_CHECK(boost::contains(fi_0.get_extra_info(), "No active child."));
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
        BOOST_CHECK(boost::contains(fi_0.get_extra_info(), "No active child."));
    }
    {
        // Test: try to generate an error in the evolution.
        // NOTE: de wants more than 1 individual in the pop.
        island fi_0(fork_island{}, de{1}, rosenbrock{}, 1);
        BOOST_CHECK(fi_0.extract<fork_island>() != nullptr);
        BOOST_CHECK(boost::contains(fi_0.get_extra_info(), "No active child."));
        fi_0.evolve();
        BOOST_CHECK_EXCEPTION(fi_0.wait_check(), std::runtime_error, [](const std::runtime_error &re) {
            return boost::contains(re.what(), "needs at least 5 individuals in the population");
        });
    }
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
    void serialize(Archive &ar)
    {
        ar(n_evolve);
    }
    mutable int n_evolve = 0;
};

PAGMO_REGISTER_ALGORITHM(stateful_algo)

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
    void serialize(Archive &)
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
    void serialize(Archive &)
    {
    }
};

PAGMO_REGISTER_ALGORITHM(recursive_algo1)
PAGMO_REGISTER_ALGORITHM(recursive_algo2)

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
    {
        // Try also error transport.
        island fi_0(fork_island{}, recursive_algo2{}, rosenbrock{}, 1, 0);
        fi_0.evolve();
        BOOST_CHECK_EXCEPTION(fi_0.wait_check(), std::runtime_error, [](const std::runtime_error &re) {
            return boost::contains(re.what(), "needs at least 5 individuals in the population");
        });
    }
}
