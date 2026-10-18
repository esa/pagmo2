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

#define BOOST_TEST_MODULE fitness_rewrite_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/bee_colony.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/algorithms/gwo.hpp>
#include <pagmo/algorithms/ihs.hpp>
#include <pagmo/algorithms/maco.hpp>
#include <pagmo/algorithms/moead.hpp>
#include <pagmo/algorithms/moead_gen.hpp>
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/algorithms/nsga3.hpp>
#include <pagmo/algorithms/nspso.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/algorithms/pso_gen.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/sea.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>
#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/config.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#if defined(PAGMO_WITH_EIGEN3)

#include <pagmo/algorithms/cmaes.hpp>
#include <pagmo/algorithms/xnes.hpp>

#endif

using namespace pagmo;

// In this file we check that the algorithms supporting in-place rewriting
// of the decision vectors (see problem::fitness_rewrite()) store the rewritten
// decision vectors in the population, and that the stored decision vectors
// are consistent with the stored fitness vectors.
//
// The test problems "decode" a decision vector by snapping it to a grid with
// step 1/8: the rewritten decision vectors are thus all on the grid, while
// the decision vectors produced by the genetic operators are (almost surely) not.

static vector_double snap(vector_double x)
{
    for (auto &xi : x) {
        xi = std::round(xi * 8.) / 8.;
    }
    return x;
}

static bool on_grid(const vector_double &x)
{
    return snap(x) == x;
}

// Single-objective problem without rewriting.
struct so_plain {
    static vector_double eval(const vector_double &x)
    {
        double retval = 0.;
        for (auto xi : x) {
            retval += (xi - .3) * (xi - .3);
        }
        return {retval};
    }
    vector_double fitness(const vector_double &x) const
    {
        return eval(snap(x));
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {vector_double(4u, -1.), vector_double(4u, 1.)};
    }
};

// Single-objective problem with rewriting.
struct so_rw : so_plain {
    vector_double fitness_rewrite(vector_double &x) const
    {
        x = snap(x);
        return eval(x);
    }
};

// Stochastic single-objective problem with rewriting.
struct so_rw_sto : so_rw {
    vector_double fitness(const vector_double &x) const
    {
        return {so_rw::fitness(x)[0] + 1e-3 * static_cast<double>(m_seed % 7u)};
    }
    vector_double fitness_rewrite(vector_double &x) const
    {
        return {so_rw::fitness_rewrite(x)[0] + 1e-3 * static_cast<double>(m_seed % 7u)};
    }
    void set_seed(unsigned seed)
    {
        m_seed = seed;
    }
    unsigned m_seed = 0u;
};

// Multi-objective (ZDT1-like) problem without rewriting.
struct mo_plain {
    static vector_double eval(const vector_double &x)
    {
        double g = 0.;
        for (decltype(x.size()) i = 1u; i < x.size(); ++i) {
            g += x[i];
        }
        g = 1. + 9. * g / static_cast<double>(x.size() - 1u);
        return {x[0], g * (1. - std::sqrt(x[0] / g))};
    }
    vector_double fitness(const vector_double &x) const
    {
        return eval(snap(x));
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {vector_double(6u, 0.), vector_double(6u, 1.)};
    }
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }
};

// Multi-objective problem with rewriting.
struct mo_rw : mo_plain {
    vector_double fitness_rewrite(vector_double &x) const
    {
        x = snap(x);
        return eval(x);
    }
};

// Multi-objective problem with rewriting and batch_fitness() (which cannot rewrite).
struct mo_rw_batch : mo_rw {
    vector_double batch_fitness(const vector_double &dvs) const
    {
        vector_double retval;
        for (decltype(dvs.size()) i = 0u; i < dvs.size(); i += 6u) {
            const auto fv = fitness(vector_double(dvs.data() + i, dvs.data() + i + 6u));
            retval.insert(retval.end(), fv.begin(), fv.end());
        }
        return retval;
    }
};

// Check that the stored fitness vectors are consistent with the stored decision vectors and,
// if expect_grid is true, that the stored decision vectors (and the champion) are on the grid.
static void check_pop(const population &pop, bool expect_grid, const std::string &name)
{
    BOOST_TEST_CONTEXT(name)
    {
        const auto &prob = pop.get_problem();
        BOOST_CHECK(pop.size() > 0u);
        for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
            BOOST_CHECK(pop.get_f()[i] == prob.fitness(pop.get_x()[i]));
            if (expect_grid) {
                BOOST_CHECK(on_grid(pop.get_x()[i]));
            }
        }
        if (prob.get_nobj() == 1u && !prob.is_stochastic() && expect_grid) {
            BOOST_CHECK(on_grid(pop.champion_x()));
            BOOST_CHECK(pop.champion_f() == prob.fitness(pop.champion_x()));
        }
    }
}

// Evolve a population of the input problem with the input algorithm, and check the result.
static population run(algorithm algo, const problem &prob, population::size_type pop_size, bool expect_grid = true)
{
    population pop{prob, pop_size, 42u};
    check_pop(pop, expect_grid || prob.has_fitness_rewrite(), algo.get_name() + " (initial population)");
    if (algo.has_set_seed()) {
        algo.set_seed(42u);
    }
    pop = algo.evolve(pop);
    check_pop(pop, expect_grid, algo.get_name());
    return pop;
}

template <typename UDA>
static UDA with_bfe(UDA uda)
{
    uda.set_bfe(bfe{thread_bfe{}});
    return uda;
}

BOOST_AUTO_TEST_CASE(single_objective)
{
    const problem prob{so_rw{}};
    BOOST_CHECK(prob.has_fitness_rewrite());
    run(algorithm{sga{10u}}, prob, 20u);
    run(algorithm{sea{10u}}, prob, 20u);
    run(algorithm{de{10u}}, prob, 20u);
    run(algorithm{with_bfe(de{10u})}, prob, 20u);
    run(algorithm{de1220{10u}}, prob, 20u);
    run(algorithm{sade{10u}}, prob, 20u);
    run(algorithm{pso{10u}}, prob, 20u);
    run(algorithm{pso_gen{10u}}, prob, 20u);
    run(algorithm{with_bfe(pso_gen{10u})}, prob, 20u);
    run(algorithm{gwo{10u}}, prob, 20u);
    run(algorithm{bee_colony{10u, 1u}}, prob, 20u);
    run(algorithm{ihs{100u}}, prob, 20u);
    run(algorithm{simulated_annealing{}}, prob, 5u);
    run(algorithm{compass_search{200u}}, prob, 5u);
    run(algorithm{gaco{10u, 10u}}, prob, 20u);
    run(algorithm{with_bfe(gaco{10u, 10u})}, prob, 20u);
}

BOOST_AUTO_TEST_CASE(single_objective_stochastic)
{
    const problem prob{so_rw_sto{}};
    BOOST_CHECK(prob.has_fitness_rewrite());
    BOOST_CHECK(prob.is_stochastic());
    run(algorithm{sga{10u}}, prob, 20u);
    run(algorithm{sea{10u}}, prob, 20u);
    run(algorithm{pso_gen{10u}}, prob, 20u);
    run(algorithm{with_bfe(pso_gen{10u})}, prob, 20u);
}

BOOST_AUTO_TEST_CASE(multi_objective)
{
    const problem prob{mo_rw{}};
    BOOST_CHECK(prob.has_fitness_rewrite());
    run(algorithm{nsga2{10u}}, prob, 20u);
    run(algorithm{with_bfe(nsga2{10u})}, prob, 20u);
    run(algorithm{nsga3{10u}}, prob, 20u);
    run(algorithm{with_bfe(nsga3{10u})}, prob, 20u);
    run(algorithm{moead{10u}}, prob, 40u);
    run(algorithm{moead_gen{10u}}, prob, 40u);
    run(algorithm{with_bfe(moead_gen{10u})}, prob, 40u);
    run(algorithm{nspso{10u}}, prob, 20u);
    run(algorithm{with_bfe(nspso{10u})}, prob, 20u);
    run(algorithm{maco{10u, 10u}}, prob, 20u);
    run(algorithm{with_bfe(maco{10u, 10u})}, prob, 20u);
    run(algorithm{ihs{100u}}, prob, 20u);
}

BOOST_AUTO_TEST_CASE(serial_bfe_equivalence)
{
    // DE and NSGA-III consume the random engine identically
    // in the serial and in the bfe paths.
    auto pop0 = run(algorithm{de{10u}}, problem{so_rw{}}, 20u);
    auto pop1 = run(algorithm{with_bfe(de{10u})}, problem{so_rw{}}, 20u);
    BOOST_CHECK(pop0.get_x() == pop1.get_x());
    BOOST_CHECK(pop0.get_f() == pop1.get_f());
    pop0 = run(algorithm{nsga3{10u}}, problem{mo_rw{}}, 20u);
    pop1 = run(algorithm{with_bfe(nsga3{10u})}, problem{mo_rw{}}, 20u);
    BOOST_CHECK(pop0.get_x() == pop1.get_x());
    BOOST_CHECK(pop0.get_f() == pop1.get_f());
}

BOOST_AUTO_TEST_CASE(no_rewrite_control)
{
    // Without rewriting, the decision vectors are not snapped to the grid
    // (but they are still consistent with the fitness vectors).
    for (auto pop : {run(algorithm{sga{10u}}, problem{so_plain{}}, 20u, false),
                     run(algorithm{nsga2{10u}}, problem{mo_plain{}}, 20u, false)}) {
        BOOST_CHECK(
            std::any_of(pop.get_x().begin(), pop.get_x().end(), [](const vector_double &x) { return !on_grid(x); }));
    }
}

BOOST_AUTO_TEST_CASE(unsupported_bfe)
{
    // A bfe which does not support rewriting raises an error.
    nsga2 uda{10u};
    uda.set_bfe(bfe{member_bfe{}});
    population pop{mo_rw_batch{}, 20u, 42u};
    BOOST_CHECK_THROW(algorithm{uda}.evolve(pop), not_implemented_error);
    // It works as usual with a problem which does not rewrite.
    population pop_plain{mo_plain{}, 20u, 42u};
    uda.set_bfe(bfe{thread_bfe{}});
    pop_plain = algorithm{uda}.evolve(pop_plain);
    check_pop(pop_plain, false, uda.get_name());
}

#if defined(PAGMO_WITH_EIGEN3)

BOOST_AUTO_TEST_CASE(distribution_based)
{
    // CMA-ES and xNES never rewrite the decision vectors they generate,
    // but the population stays consistent.
    for (auto pop : {run(algorithm{cmaes{10u}}, problem{so_rw{}}, 20u, false),
                     run(algorithm{with_bfe(cmaes{10u})}, problem{so_rw{}}, 20u, false),
                     run(algorithm{xnes{10u}}, problem{so_rw{}}, 20u, false)}) {
        BOOST_CHECK(
            std::any_of(pop.get_x().begin(), pop.get_x().end(), [](const vector_double &x) { return !on_grid(x); }));
    }
}

#endif
