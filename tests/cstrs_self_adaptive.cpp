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

#define BOOST_TEST_MODULE cstrs_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <cmath>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/cstrs_self_adaptive.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/cec2006.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(penalized_problem_construction)
{
    using namespace detail;
    auto NP = 20u;
    problem udp{cec2006{1u}};
    population pop{udp, NP};
    penalized_udp udp_p{pop};
    BOOST_CHECK(udp_p.m_pop_ptr == &pop);
    BOOST_CHECK_EQUAL(udp_p.m_c_max.size(), udp.get_nc());
    BOOST_CHECK_EQUAL(udp_p.m_f_hat_down.size(), udp.get_nf());
    BOOST_CHECK_EQUAL(udp_p.m_f_hat_up.size(), udp.get_nf());
    BOOST_CHECK_EQUAL(udp_p.m_f_hat_round.size(), udp.get_nf());
    BOOST_CHECK_EQUAL(udp_p.m_fitness_map.size(), NP);
    // We also test get bounds here
    BOOST_CHECK(udp_p.get_bounds() == udp.get_bounds());
    // And the debug stream operator
    std::ostringstream text;
    text << udp_p;
    BOOST_CHECK(text.str().find("Best (hat down)") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(penalized_problem_fitness_cache)
{
    using namespace detail;
    auto NP = 20u;
    problem udp{cec2006{1u}};
    population pop{udp, NP};
    penalized_udp udp_p{pop};
    BOOST_CHECK_EQUAL(udp_p.m_pop_ptr->get_problem().get_fevals(), NP);
    population new_pop{udp_p};
    // The following lines do not cause fevals increments as the cache is hit.
    for (decltype(NP) i = 0u; i < NP; ++i) {
        new_pop.push_back(pop.get_x()[i]);
    }
    // We check the cache was hit -> not increasing the fevals
    BOOST_CHECK_EQUAL(udp_p.m_pop_ptr->get_problem().get_fevals(), NP);
    new_pop.set_x(0, vector_double(13, 0.5));
    // We check the cache was not hit -> increasing the fevals
    BOOST_CHECK_EQUAL(udp_p.m_pop_ptr->get_problem().get_fevals(), NP + 1);
    new_pop.set_x(1, vector_double(13, 0.5));
    // We check the cache was hit -> not increasing the fevals
    BOOST_CHECK_EQUAL(udp_p.m_pop_ptr->get_problem().get_fevals(), NP + 1);
}

BOOST_AUTO_TEST_CASE(cstrs_self_adaptive_construction)
{
    { // default constructor
        cstrs_self_adaptive udp;
        BOOST_CHECK(udp.get_inner_algorithm().extract<de>() != NULL);
        BOOST_CHECK(udp.get_inner_algorithm().extract<compass_search>() == NULL);
    }
    { // constructor from iters
        BOOST_CHECK_NO_THROW((cstrs_self_adaptive{1500u}));
        BOOST_CHECK_NO_THROW((cstrs_self_adaptive{1500u, de{}}));
        BOOST_CHECK_NO_THROW((cstrs_self_adaptive{1500u, de{}, 32u}));
    }
    // Here we only test that evolution is deterministic if the
    // seed is controlled
    {
        problem prob{hock_schittkowsky_71{}};
        prob.set_c_tol({1e-3, 1e-3});
        population pop1{prob, 5u, 23u};
        population pop2{prob, 5u, 23u};
        population pop3{prob, 5u, 23u};

        cstrs_self_adaptive user_algo1{150u, de{10u, 0.8, 0.9, 2u, 1e-6, 1e-6, 32u}, 32u};
        user_algo1.set_verbosity(1u);
        pop1 = user_algo1.evolve(pop1);
        BOOST_CHECK(user_algo1.get_log().size() > 0u);

        cstrs_self_adaptive user_algo2{150u, de{10u, 0.8, 0.9, 2u, 1e-6, 1e-6, 32u}, 32u};
        user_algo2.set_verbosity(1u);
        pop2 = user_algo2.evolve(pop2);

        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());
        user_algo2.set_seed(32u);
        user_algo2.get_inner_algorithm().extract<de>()->set_seed(32u);
        pop3 = user_algo2.evolve(pop3);

        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());
    }
    // We then check that the evolve throws if called on unsuitable problems
    {
        cstrs_self_adaptive user_algo{150u, de{10u, 0.8, 0.9, 2u, 1e-6, 1e-6, 32u}, 32u};
        BOOST_CHECK_THROW(user_algo.evolve(population{zdt{}, 15u}), std::invalid_argument);
    }
    {
        cstrs_self_adaptive user_algo{150u, de{10u, 0.8, 0.9, 2u, 1e-6, 1e-6, 32u}, 32u};
        BOOST_CHECK_THROW(user_algo.evolve(population{inventory{}, 15u}), std::invalid_argument);
    }
    {
        cstrs_self_adaptive user_algo{150u, de{10u, 0.8, 0.9, 2u, 1e-6, 1e-6, 32u}, 32u};
        BOOST_CHECK_THROW(user_algo.evolve(population{rosenbrock{}, 15u}), std::invalid_argument);
    }
    {
        cstrs_self_adaptive user_algo{150u, de{10u, 0.8, 0.9, 2u, 1e-6, 1e-6, 32u}, 32u};
        BOOST_CHECK_THROW(user_algo.evolve(population{hock_schittkowsky_71{}, 3u}), std::invalid_argument);
    }
    // And a clean exit for 0 iterations
    problem prob{hock_schittkowsky_71{}};
    population pop{prob, 10u};
    BOOST_CHECK((cstrs_self_adaptive{0u, de{10u, 0.8, 0.9, 2u, 1e-6, 1e-6, 32u}, 32u}.evolve(pop).get_x()[0])
                == (pop.get_x()[0]));
}

BOOST_AUTO_TEST_CASE(cstrs_self_adaptive_serialization)
{
    // Make one evolution
    problem prob{hock_schittkowsky_71{}};
    population pop{prob, 10u, 23u};
    algorithm algo{cstrs_self_adaptive{1500u, de{1u, 0.8, 0.9, 2u, 1e-6, 1e-6, 32u}, 32u}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<cstrs_self_adaptive>()->get_log();
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
    auto after_log = algo.extract<cstrs_self_adaptive>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    BOOST_CHECK(before_log == after_log);
    // so we implement a close check
    BOOST_CHECK(before_log.size() > 0u);
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_EQUAL(std::get<1>(before_log[i]), std::get<1>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<2>(before_log[i]), std::get<2>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<3>(before_log[i]), std::get<3>(after_log[i]), 1e-8);
        BOOST_CHECK_EQUAL(std::get<4>(before_log[i]), std::get<4>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<5>(before_log[i]), std::get<5>(after_log[i]), 1e-8);
        BOOST_CHECK_EQUAL(std::get<6>(before_log[i]), std::get<6>(after_log[i]));
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

BOOST_AUTO_TEST_CASE(cstrs_self_adaptive_threading_test)
{
    BOOST_CHECK((algorithm{cstrs_self_adaptive{1500u, ts1{}, 32u}}.get_thread_safety() == thread_safety::basic));
    BOOST_CHECK((algorithm{cstrs_self_adaptive{1500u, ts2{}, 32u}}.get_thread_safety() == thread_safety::none));
    BOOST_CHECK((algorithm{cstrs_self_adaptive{1500u, ts3{}, 32u}}.get_thread_safety() == thread_safety::basic));
}

struct ia1 {
    population evolve(const population &pop) const
    {
        return pop;
    }
    double m_data = 0.;
};

BOOST_AUTO_TEST_CASE(cstrs_self_adaptive_inner_algo_get_test)
{
    // We check that the correct overload is called according to (*this) being const or not
    {
        const cstrs_self_adaptive uda(1500u, ia1{}, 32u);
        BOOST_CHECK(std::is_const<decltype(uda)>::value);
        BOOST_CHECK(std::is_const<std::remove_reference<decltype(uda.get_inner_algorithm())>::type>::value);
    }
    {
        cstrs_self_adaptive uda(1500u, ia1{}, 32u);
        BOOST_CHECK(!std::is_const<decltype(uda)>::value);
        BOOST_CHECK(!std::is_const<std::remove_reference<decltype(uda.get_inner_algorithm())>::type>::value);
    }
}

BOOST_AUTO_TEST_CASE(cstrs_self_adaptive_all_feasible_test)
{
    // We check the behavior when all individuals are feasible

    // We build an all feasible population
    problem prob{hock_schittkowsky_71{}};
    population pop{prob, 4u};
    BOOST_CHECK_EQUAL(prob.get_nf(), 3u);
    BOOST_CHECK_EQUAL(prob.get_nobj(), 1u);
    BOOST_CHECK_EQUAL(prob.get_nec(), 1u);
    BOOST_CHECK_EQUAL(prob.get_nic(), 1u);
    pop.set_xf(0, vector_double{1., 2., 3., 4.}, vector_double{2, 0., -1});
    pop.set_xf(1, vector_double{2., 2., 3., 4.}, vector_double{3, 0., -2});
    pop.set_xf(2, vector_double{3., 2., 3., 4.}, vector_double{5, 0., -2});
    pop.set_xf(3, vector_double{4., 2., 3., 4.}, vector_double{7, 0., -23});

    // We define the penalized_udp problem. Upon construction the update method will be called
    // and all penalties assigned. We test the correctness of their value
    detail::penalized_udp udp_p{pop};
    BOOST_CHECK_EQUAL(udp_p.m_scaling_factor, 0.);
    BOOST_CHECK_EQUAL(udp_p.m_i_hat_up, 0.);
    BOOST_CHECK_EQUAL(udp_p.m_i_hat_down, 0.);
    BOOST_CHECK_EQUAL(udp_p.m_i_hat_round, 0.);
    BOOST_CHECK_EQUAL(udp_p.m_apply_penalty_1, false);
    BOOST_CHECK((udp_p.m_c_max == vector_double{0., 0.}));
    BOOST_CHECK_EQUAL(udp_p.m_n_feasible, 4);
    BOOST_CHECK((udp_p.m_f_hat_up == vector_double{2, 0., -1}));
    BOOST_CHECK((udp_p.m_f_hat_round == vector_double{2, 0., -1}));
}

BOOST_AUTO_TEST_CASE(cstrs_self_adaptive_infeasible_better_than_hat_down_test)
{
    // We check the behavior when all individuals are feasible

    // We build an all feasible population
    problem prob{hock_schittkowsky_71{}};
    population pop{prob, 4u};
    BOOST_CHECK_EQUAL(prob.get_nf(), 3u);
    BOOST_CHECK_EQUAL(prob.get_nobj(), 1u);
    BOOST_CHECK_EQUAL(prob.get_nec(), 1u);
    BOOST_CHECK_EQUAL(prob.get_nic(), 1u);
    pop.set_xf(0, vector_double{1., 2., 3., 4.}, vector_double{2, 0., -1.}); // feasible (hat_down)
    pop.set_xf(1, vector_double{2., 2., 3., 4.}, vector_double{1, 0., 1.});  // infeasible (better than hat_down)
    pop.set_xf(2, vector_double{2., 2., 3., 4.}, vector_double{0, 0., 1.});  // infeasible (better than hat_down)
    pop.set_xf(3, vector_double{4., 2., 3., 4.}, vector_double{7, 3.4, 23});

    // We define the penalized_udp problem. Upon construction the update method will be called
    // and all penalties assigned. We test the correctness of their value
    detail::penalized_udp udp_p{pop};
    BOOST_CHECK((udp_p.m_f_hat_up == vector_double{0, 0., 1.}));
    BOOST_CHECK((udp_p.m_f_hat_down == vector_double{2, 0., -1.}));
    BOOST_CHECK((udp_p.m_f_hat_round == vector_double{7, 3.4, 23}));
    BOOST_CHECK_EQUAL(udp_p.m_apply_penalty_1, true);
}

BOOST_AUTO_TEST_CASE(cstrs_self_adaptive_infeasible_not_better_than_hat_down_test)
{
    // We check the behavior when all individuals are feasible

    // We build an all feasible population
    problem prob{hock_schittkowsky_71{}};
    population pop{prob, 5u};
    BOOST_CHECK_EQUAL(prob.get_nf(), 3u);
    BOOST_CHECK_EQUAL(prob.get_nobj(), 1u);
    BOOST_CHECK_EQUAL(prob.get_nec(), 1u);
    BOOST_CHECK_EQUAL(prob.get_nic(), 1u);
    pop.set_xf(0, vector_double{1., 2., 3., 4.}, vector_double{2, 0., -1.});  // feasible (hat_down)
    pop.set_xf(1, vector_double{2., 2., 3., 4.}, vector_double{3, 21, 10.});  // infeasible (worse than hat_down)
    pop.set_xf(2, vector_double{2., 2., 3., 4.}, vector_double{3, 22, 10.});  // infeasible (worse than hat_down)
    pop.set_xf(3, vector_double{2., 2., 3., 4.}, vector_double{4, 22., 10.}); // infeasible (worse than hat_down)
    pop.set_xf(4, vector_double{4., 2., 3., 4.}, vector_double{7, 3.4, 23});

    // We define the penalized_udp problem. Upon construction the update method will be called
    // and all penalties assigned. We test the correctness of their value
    detail::penalized_udp udp_p{pop};
    BOOST_CHECK((udp_p.m_f_hat_up == vector_double{4, 22., 10.}));
    BOOST_CHECK((udp_p.m_f_hat_down == vector_double{2, 0., -1.}));
    BOOST_CHECK((udp_p.m_f_hat_round == vector_double{7, 3.4, 23}));
    BOOST_CHECK_EQUAL(udp_p.m_apply_penalty_1, false);
}

BOOST_AUTO_TEST_CASE(cstrs_self_adaptive_all_infeasible_test)
{
    // We check the behavior when all individuals are feasible

    // We build an all feasible population
    problem prob{hock_schittkowsky_71{}};
    population pop{prob, 4u};
    BOOST_CHECK_EQUAL(prob.get_nf(), 3u);
    BOOST_CHECK_EQUAL(prob.get_nobj(), 1u);
    BOOST_CHECK_EQUAL(prob.get_nec(), 1u);
    BOOST_CHECK_EQUAL(prob.get_nic(), 1u);
    pop.set_xf(0, vector_double{1., 2., 3., 4.}, vector_double{2., 1., 1.});
    pop.set_xf(1, vector_double{2., 2., 3., 4.}, vector_double{1., 1., 1.});
    pop.set_xf(2, vector_double{2., 2., 3., 4.}, vector_double{4, 22., 10.});
    pop.set_xf(3, vector_double{4., 2., 3., 4.}, vector_double{7, 22., 10.});

    // We define the penalized_udp problem. Upon construction the update method will be called
    // and all penalties assigned. We test the correctness of their value
    detail::penalized_udp udp_p{pop};
    BOOST_CHECK((udp_p.m_f_hat_up == vector_double{7, 22., 10.}));
    BOOST_CHECK((udp_p.m_f_hat_down == vector_double{1., 1., 1.}));
    BOOST_CHECK((udp_p.m_f_hat_round == vector_double{7, 22., 10.}));
    BOOST_CHECK_EQUAL(udp_p.m_apply_penalty_1, true);
}