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

#include <pagmo/problem.hpp>

#define BOOST_TEST_MODULE problem_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

// Generates a dummy simple problem with arbitrary dimensions and return values
struct base_p {
    base_p(vector_double::size_type nobj = 1u, vector_double::size_type nec = 0u, vector_double::size_type nic = 0u,
           const vector_double &ret_fit = {1.}, const vector_double &lb = {0.}, const vector_double &ub = {1.})
        : m_nobj(nobj), m_nec(nec), m_nic(nic), m_ret_fit(ret_fit), m_lb(lb), m_ub(ub)
    {
    }

    vector_double fitness(const vector_double &) const
    {
        return m_ret_fit;
    }
    vector_double::size_type get_nobj() const
    {
        return m_nobj;
    }
    vector_double::size_type get_nec() const
    {
        return m_nec;
    }
    vector_double::size_type get_nic() const
    {
        return m_nic;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {m_lb, m_ub};
    }
    std::string get_name() const
    {
        return "A base toy problem";
    }

    std::string get_extra_info() const
    {
        return "Nothing to report";
    }

    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_nobj, m_nec, m_nic, m_ret_fit, m_lb, m_ub);
    }

    vector_double::size_type m_nobj;
    vector_double::size_type m_nec;
    vector_double::size_type m_nic;
    vector_double m_ret_fit;
    vector_double m_lb;
    vector_double m_ub;
};

// Generates a dummy problem with arbitrary dimensions and return values
// having the gradient implemented
struct grad_p : base_p {
    grad_p(vector_double::size_type nobj = 1u, vector_double::size_type nec = 0u, vector_double::size_type nic = 0u,
           const vector_double &ret_fit = {1}, const vector_double &lb = {0}, const vector_double &ub = {1},
           const vector_double &g = {1}, const sparsity_pattern &gs = {{0, 0}})
        : base_p(nobj, nec, nic, ret_fit, lb, ub), m_g(g), m_gs(gs)
    {
    }

    vector_double gradient(const vector_double &) const
    {
        return m_g;
    }

    sparsity_pattern gradient_sparsity() const
    {
        return m_gs;
    }

    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<base_p>(this), m_g, m_gs);
    }

    vector_double m_g;
    sparsity_pattern m_gs;
};

PAGMO_REGISTER_PROBLEM(grad_p)

// Generates a dummy problem with arbitrary dimensions and return values
// having the gradient implemented but overriding the has methods
struct grad_p_override : grad_p {
    grad_p_override(vector_double::size_type nobj = 1u, vector_double::size_type nec = 0u,
                    vector_double::size_type nic = 0u, const vector_double &ret_fit = {1},
                    const vector_double &lb = {0}, const vector_double &ub = {1}, const vector_double &g = {1},
                    const sparsity_pattern &gs = {{0, 0}})
        : grad_p(nobj, nec, nic, ret_fit, lb, ub, g, gs)
    {
    }

    bool has_gradient() const
    {
        return false;
    }

    bool has_gradient_sparsity() const
    {
        return false;
    }

    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<grad_p>(this));
    }
};

PAGMO_REGISTER_PROBLEM(grad_p_override)

// Generates a dummy problem with arbitrary dimensions and return values
// having the hessians implemented
struct hess_p : base_p {
    hess_p(vector_double::size_type nobj = 1u, vector_double::size_type nec = 0u, vector_double::size_type nic = 0u,
           const vector_double &ret_fit = {1}, const vector_double &lb = {0}, const vector_double &ub = {1},
           const std::vector<vector_double> &h = {{1}}, const std::vector<sparsity_pattern> &hs = {{{0, 0}}})
        : base_p(nobj, nec, nic, ret_fit, lb, ub), m_h(h), m_hs(hs)
    {
    }

    std::vector<vector_double> hessians(const vector_double &) const
    {
        return m_h;
    }

    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return m_hs;
    }

    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<base_p>(this), m_h, m_hs);
    }

    std::vector<vector_double> m_h;
    std::vector<sparsity_pattern> m_hs;
};

PAGMO_REGISTER_PROBLEM(hess_p)

// Generates a dummy problem with arbitrary dimensions and return values
// having the hessians implemented but overriding the has methods
struct hess_p_override : hess_p {
    hess_p_override(vector_double::size_type nobj = 1u, vector_double::size_type nec = 0u,
                    vector_double::size_type nic = 0u, const vector_double &ret_fit = {1},
                    const vector_double &lb = {0}, const vector_double &ub = {1},
                    const std::vector<vector_double> &h = {{1}}, const std::vector<sparsity_pattern> &hs = {{{0, 0}}})
        : hess_p(nobj, nec, nic, ret_fit, lb, ub, h, hs)
    {
    }

    bool has_hessians() const
    {
        return false;
    }

    bool has_hessians_sparsity() const
    {
        return false;
    }

    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<hess_p>(this));
    }
};

PAGMO_REGISTER_PROBLEM(hess_p_override)

// Generates a dummy problem with arbitrary dimensions and return values
// having the hessians and the gradients implemented
struct full_p : grad_p {
    full_p(vector_double::size_type nobj = 1u, vector_double::size_type nec = 0u, vector_double::size_type nic = 0u,
           const vector_double &ret_fit = {1}, const vector_double &lb = {0}, const vector_double &ub = {1},
           const vector_double &g = {1}, const sparsity_pattern &gs = {{0, 0}},
           const std::vector<vector_double> &h = {{1}}, const std::vector<sparsity_pattern> &hs = {{{0, 0}}})
        : grad_p(nobj, nec, nic, ret_fit, lb, ub, g, gs), m_h(h), m_hs(hs)
    {
    }

    std::vector<vector_double> hessians(const vector_double &) const
    {
        return m_h;
    }

    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return m_hs;
    }

    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<grad_p>(this), m_h, m_hs);
    }

    std::vector<vector_double> m_h;
    std::vector<sparsity_pattern> m_hs;
};

PAGMO_REGISTER_PROBLEM(full_p)

struct empty {
    vector_double fitness(const vector_double &) const
    {
        return {1};
    }
    vector_double::size_type get_nec() const
    {
        return 0;
    }
    vector_double::size_type get_nic() const
    {
        return 0;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
};

BOOST_AUTO_TEST_CASE(problem_construction_test)
{
    // We check that problems with inconsistent dimensions throw
    // std::invalid argument
    vector_double lb_2(2, 0);
    vector_double ub_2(2, 1);
    vector_double lb_3(3, 1);
    vector_double ub_3(3, 1);
    vector_double fit_1(1, 1);
    vector_double fit_2(2, 1);
    vector_double fit_12(12, 11);
    vector_double lb_11(11, 0);
    vector_double ub_11(11, 0);

    vector_double grad_2{1, 1};
    sparsity_pattern grads_2_outofbounds{{0, 0}, {3, 4}};
    sparsity_pattern grads_2_repeats{{0, 0}, {0, 0}};
    sparsity_pattern grads_2_correct{{0, 0}, {0, 1}};

    std::vector<vector_double> hess_22{{1, 1}, {1, 1}};
    std::vector<sparsity_pattern> hesss_22_outofbounds{{{0, 0}, {12, 13}}, {{0, 0}, {1, 0}}};
    std::vector<sparsity_pattern> hesss_22_notlowertriangular{{{0, 0}, {0, 1}}, {{0, 0}, {1, 0}}};
    std::vector<sparsity_pattern> hesss_22_repeated{{{0, 0}, {0, 0}}, {{0, 0}, {1, 0}}};
    std::vector<sparsity_pattern> hesss_22_correct{{{0, 0}, {1, 0}}, {{0, 0}, {1, 0}}};

    // 0 - lb size is zero
    BOOST_CHECK_THROW(problem{base_p(1, 0, 0, fit_1, {}, {})}, std::invalid_argument);
    // 1 - lb > ub
    BOOST_CHECK_THROW(problem{base_p(1, 0, 0, fit_1, ub_2, lb_2)}, std::invalid_argument);
    // 2 - lb length is wrong
    BOOST_CHECK_THROW(problem{base_p(1, 0, 0, fit_1, lb_3, ub_2)}, std::invalid_argument);
    // 3 - ub length is wrong
    BOOST_CHECK_THROW(problem{base_p(1, 0, 0, fit_1, lb_2, ub_3)}, std::invalid_argument);
    // 4 - gradient sparsity has index out of bounds
    BOOST_CHECK_THROW(problem{grad_p(1, 0, 0, fit_1, lb_2, ub_2, grad_2, grads_2_outofbounds)}, std::invalid_argument);
    // 5 - gradient sparsity has a repeating pair
    BOOST_CHECK_THROW(problem{grad_p(1, 0, 0, fit_1, lb_2, ub_2, grad_2, grads_2_repeats)}, std::invalid_argument);
    // 6 - hessian sparsity has index out of bounds
    BOOST_CHECK_THROW(problem{hess_p(1, 1, 0, fit_2, lb_2, ub_2, hess_22, hesss_22_outofbounds)},
                      std::invalid_argument);
    // 7 - hessian sparsity is not lower triangular
    BOOST_CHECK_THROW(problem{hess_p(1, 1, 0, fit_2, lb_2, ub_2, hess_22, hesss_22_notlowertriangular)},
                      std::invalid_argument);
    // 8 - hessian sparsity has repeated indexes
    BOOST_CHECK_THROW(problem{hess_p(1, 1, 0, fit_2, lb_2, ub_2, hess_22, hesss_22_repeated)}, std::invalid_argument);
    // 9 - hessian sparsity has the wrong length
    BOOST_CHECK_THROW(
        problem{hess_p(1, 1, 0, fit_2, lb_2, ub_2, hess_22, {{{0, 0}, {1, 0}}, {{0, 0}, {1, 0}}, {{0, 0}}})},
        std::invalid_argument);
    // 10 - 0 objectives
    BOOST_CHECK_THROW(problem{base_p(0, 0, 0, fit_1, {1}, {2})}, std::invalid_argument);
    // 11 - many objectives
    BOOST_CHECK_THROW(problem{base_p(std::numeric_limits<vector_double::size_type>::max(), 0, 0, fit_2, {1}, {2})},
                      std::invalid_argument);
    // 12 - too many equalities
    BOOST_CHECK_THROW(problem{base_p(1, std::numeric_limits<vector_double::size_type>::max(), 0, fit_2, {1}, {2})},
                      std::invalid_argument);
    // 13 - too many inequalities
    BOOST_CHECK_THROW(problem{base_p(1, 0, std::numeric_limits<vector_double::size_type>::max(), fit_2, {1}, {2})},
                      std::invalid_argument);
    // We check that the data members are initialized correctly (i.e. counters to zero
    // and gradient / hessian dimensions to the right values
    {
        problem p1{base_p(2, 0, 0, fit_2, lb_2, ub_2)};
        problem p2{base_p(3, 4, 5, fit_12, lb_11, ub_11)};
        problem p3{grad_p(1, 0, 0, fit_2, lb_2, ub_2, grad_2, grads_2_correct)};
        problem p4{hess_p(1, 1, 0, fit_2, lb_2, ub_2, hess_22, hesss_22_correct)};
        BOOST_CHECK(p1.get_fevals() == 0u);
        BOOST_CHECK(p1.get_gevals() == 0u);
        BOOST_CHECK(p1.get_hevals() == 0u);
        BOOST_CHECK(p1.get_hevals() == 0u);
    }

    // We check the move constructor
    {
        problem p1{full_p(2, 0, 0, fit_2, lb_2, ub_2, grad_2, grads_2_correct, hess_22, hesss_22_correct)};

        // We increment the counters so that the default values are changed
        p1.fitness({1, 1});
        p1.gradient({1, 1});
        p1.hessians({1, 1});

        auto p1_string = boost::lexical_cast<std::string>(p1);
        auto a1 = p1.extract<full_p>();

        problem p2(std::move(p1));

        auto a2 = p2.extract<full_p>();
        auto p2_string = boost::lexical_cast<std::string>(p2);

        // 1 - We check the resource pointed by m_ptr has been moved from p1 to p2
        BOOST_CHECK(a1 == a2);
        // 2 - We check that the two outputs of human_readable are identical
        BOOST_CHECK(p1_string == p2_string);
    }

    // We check the copy constructor
    {
        problem p1{full_p(2, 0, 0, fit_2, lb_2, ub_2, grad_2, grads_2_correct, hess_22, hesss_22_correct)};

        // We increment the counters so that the default values are changed
        p1.fitness({1, 1});
        p1.gradient({1, 1});
        p1.hessians({1, 1});

        auto a1 = p1.extract<full_p>();

        problem p2(p1);

        auto a2 = p2.extract<full_p>();

        // 1 - We check the resource pointed by m_ptr has a different addres
        BOOST_CHECK(a1 != 0);
        BOOST_CHECK(a2 != 0);
        BOOST_CHECK(a1 != a2);
        // 2 - We check that the counters are maintained by the copy operation
        BOOST_CHECK(p2.get_fevals() == 1u);
        BOOST_CHECK(p2.get_gevals() == 1u);
        BOOST_CHECK(p2.get_hevals() == 1u);
        // 3 - We check that the decision vector dimension is copied
        BOOST_CHECK(p2.get_nx() == p1.get_nx());
    }

    // Default constructor.
    problem p0;
    BOOST_CHECK((p0.extract<null_problem>() != nullptr));
    // Check copy semantics.
    problem p1{p0};
    BOOST_CHECK((p0.extract<null_problem>() != nullptr));
    BOOST_CHECK((p1.extract<null_problem>() != nullptr));
    problem p2{full_p{}};
    p2 = p1;
    BOOST_CHECK((p2.extract<null_problem>() != nullptr));
    BOOST_CHECK((p1.extract<null_problem>() != nullptr));
    // Move semantics.
    problem p3{std::move(p0)};
    BOOST_CHECK((p3.extract<null_problem>() != nullptr));
    problem p4{full_p{}};
    p4 = std::move(p2);
    BOOST_CHECK((p4.extract<null_problem>() != nullptr));
    // Check we can revive moved-from objects.
    p0 = p4;
    BOOST_CHECK((p0.extract<null_problem>() != nullptr));
    p2 = std::move(p4);
    BOOST_CHECK((p2.extract<null_problem>() != nullptr));

    // Check the is_udp type trait.
    BOOST_CHECK(is_udp<base_p>::value);
    BOOST_CHECK(is_udp<grad_p>::value);
    BOOST_CHECK(is_udp<hess_p>::value);
    BOOST_CHECK(!is_udp<hess_p &>::value);
    BOOST_CHECK(!is_udp<const hess_p &>::value);
    BOOST_CHECK(!is_udp<const hess_p>::value);
    BOOST_CHECK(!is_udp<int>::value);
    BOOST_CHECK(!is_udp<void>::value);
    BOOST_CHECK(!is_udp<std::string>::value);
    BOOST_CHECK((std::is_constructible<problem, base_p>::value));
    BOOST_CHECK((std::is_constructible<problem, grad_p>::value));
    BOOST_CHECK((std::is_constructible<problem, hess_p>::value));
    BOOST_CHECK((std::is_constructible<problem, hess_p &>::value));
    BOOST_CHECK((std::is_constructible<problem, const hess_p &>::value));
    BOOST_CHECK((std::is_constructible<problem, hess_p &&>::value));
    BOOST_CHECK((!std::is_constructible<problem, int>::value));
    BOOST_CHECK((!std::is_constructible<problem, std::string>::value));
}

BOOST_AUTO_TEST_CASE(problem_assignment_test)
{
    vector_double lb_2(2, 0);
    vector_double ub_2(2, 1);
    vector_double fit_2(2, 1);
    vector_double grad_2{1, 1};
    sparsity_pattern grads_2_correct{{0, 0}, {0, 1}};
    std::vector<vector_double> hess_22{{1, 1}, {1, 1}};
    std::vector<sparsity_pattern> hesss_22_correct{{{0, 0}, {1, 0}}, {{0, 0}, {1, 0}}};

    // We check the move assignment
    {
        problem p1{full_p(2, 0, 0, fit_2, lb_2, ub_2, grad_2, grads_2_correct, hess_22, hesss_22_correct)};

        // We increment the counters so that the default values are changed
        p1.fitness({1, 1});
        p1.gradient({1, 1});
        p1.hessians({1, 1});

        auto p1_string = boost::lexical_cast<std::string>(p1);
        auto a1 = p1.extract<full_p>();

        problem p2{base_p{}};
        p2 = std::move(p1);

        auto a2 = p2.extract<full_p>();
        auto p2_string = boost::lexical_cast<std::string>(p2);

        // 1 - We check the resource pointed by m_ptr has been moved from p1 to p2
        BOOST_CHECK(a1 == a2);
        // 2 - We check that the two outputs of human_readable are identical
        BOOST_CHECK(p1_string == p2_string);
    }

    // We check the copy assignment
    {
        problem p1{full_p(2, 0, 0, fit_2, lb_2, ub_2, grad_2, grads_2_correct, hess_22, hesss_22_correct)};

        // We increment the counters so that the default values are changed
        p1.fitness({1, 1});
        p1.gradient({1, 1});
        p1.hessians({1, 1});

        auto a1 = p1.extract<full_p>();

        problem p2{base_p{}};
        p2 = p1;

        auto a2 = p2.extract<full_p>();

        // 1 - We check the resource pointed by m_ptr has a different addres
        BOOST_CHECK(a1 != 0);
        BOOST_CHECK(a2 != 0);
        BOOST_CHECK(a1 != a2);
        // 2 - We check that the counters are reset by the copy operation
        BOOST_CHECK(p2.get_fevals() == 1u);
        BOOST_CHECK(p2.get_gevals() == 1u);
        BOOST_CHECK(p2.get_hevals() == 1u);
        // 3 - We check that the decision vector dimension is copied
        BOOST_CHECK(p2.get_nx() == p1.get_nx());
    }
}

BOOST_AUTO_TEST_CASE(problem_extract_is_test)
{
    problem p1{base_p{2, 2, 2, {1, 1}, {5, 5}, {10, 10}}};
    auto user_problem = p1.extract<base_p>();

    // We check we have access to public data members
    BOOST_CHECK(user_problem->m_nobj == 2);
    BOOST_CHECK(user_problem->m_nec == 2);
    BOOST_CHECK(user_problem->m_nic == 2);
    BOOST_CHECK((user_problem->m_ret_fit == vector_double{1, 1}));
    BOOST_CHECK((user_problem->m_lb == vector_double{5, 5}));
    BOOST_CHECK((user_problem->m_ub == vector_double{10, 10}));

    // We check that a non succesfull cast returns a null pointer
    BOOST_CHECK(!p1.extract<full_p>());

    // We check the is method
    BOOST_CHECK(p1.is<base_p>());
    BOOST_CHECK(!p1.is<full_p>());
}

BOOST_AUTO_TEST_CASE(problem_fitness_test)
{
    problem p1{base_p{2, 2, 2, {12, 13, 14, 15, 16, 17}, {5, 5}, {10, 10}}};
    problem p1_wrong_retval{base_p{2, 2, 2, {1, 1, 1}, {5, 5}, {10, 10}}};

    // We check the fitness checks
    BOOST_CHECK_THROW(p1.fitness({3, 3, 3, 3}), std::invalid_argument);
    BOOST_CHECK_THROW(p1_wrong_retval.fitness({3, 3}), std::invalid_argument);
    // We check the fitness returns the correct value
    BOOST_CHECK((p1.fitness({3, 3}) == vector_double{12, 13, 14, 15, 16, 17}));
}

BOOST_AUTO_TEST_CASE(problem_gradient_test)
{
    problem p1{grad_p{1, 0, 0, {12}, {5, 5}, {10, 10}, {12, 13}, {{0, 0}, {0, 1}}}};
    problem p1_wrong_retval{grad_p{1, 0, 0, {12}, {5, 5}, {10, 10}, {1, 2, 3, 4}}};
    // We check the gradient checks
    BOOST_CHECK_THROW(p1.gradient({3, 3, 3}), std::invalid_argument);
    BOOST_CHECK_THROW(p1_wrong_retval.gradient({3, 3}), std::invalid_argument);
    // We check the fitness returns the correct value
    BOOST_CHECK((p1.gradient({3, 3}) == vector_double{12, 13}));

    {
        problem p2{base_p{2, 2, 2, {12, 13, 14, 15, 16, 17}, {5, 5}, {10, 10}}};
        BOOST_CHECK_THROW(p2.gradient({3, 3}), not_implemented_error);
        BOOST_CHECK_THROW(p2.hessians({3, 3}), not_implemented_error);
    }
}

BOOST_AUTO_TEST_CASE(problem_hessians_test)
{
    problem p1{hess_p{1, 0, 0, {12}, {5, 5}, {10, 10}, {{12, 13}}, {{{0, 0}, {1, 0}}}}};
    problem p1_wrong_retval{hess_p{1, 0, 0, {12}, {5, 5}, {10, 10}, {{12, 13, 14}}, {{{0, 0}, {1, 0}}}}};
    // We check the gradient checks
    BOOST_CHECK_THROW(p1.hessians({3, 3, 3}), std::invalid_argument);
    BOOST_CHECK_THROW(p1_wrong_retval.hessians({3, 3}), std::invalid_argument);
    // We check the fitness returns the correct value
    BOOST_CHECK((p1.hessians({3, 3}) == std::vector<vector_double>{{12, 13}}));
}

// We add a problem signalling gradient_sparsity() as present, but not implementing it
struct hgs_not_impl {
    vector_double fitness(const vector_double &) const
    {
        return {1., 1.};
    }
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }
    bool has_gradient_sparsity() const
    {
        return true;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
};

// We add a problem signalling hessians_sparsity() as present, but not implementing it
struct hhs_not_impl {
    vector_double fitness(const vector_double &) const
    {
        return {1., 1.};
    }
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }
    bool has_hessians_sparsity() const
    {
        return true;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
};

// We add a problem signalling set_seed() as present, but not implementing it
struct ss_not_impl {
    vector_double fitness(const vector_double &) const
    {
        return {1., 1.};
    }
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }
    bool has_set_seed() const
    {
        return true;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
};

BOOST_AUTO_TEST_CASE(problem_has_test)
{
    problem p1{base_p{}};
    problem p2{grad_p{}};
    problem p3{hess_p{}};
    problem p4{grad_p_override{}};
    problem p5{hess_p_override{}};

    BOOST_CHECK(!p1.has_gradient());
    BOOST_CHECK(!p1.has_gradient_sparsity());
    BOOST_CHECK(!p1.has_hessians());
    BOOST_CHECK(!p1.has_hessians_sparsity());

    BOOST_CHECK(p2.has_gradient());
    BOOST_CHECK(p2.has_gradient_sparsity());
    BOOST_CHECK(!p2.has_hessians());
    BOOST_CHECK(!p2.has_hessians_sparsity());

    BOOST_CHECK(!p3.has_gradient());
    BOOST_CHECK(!p3.has_gradient_sparsity());
    BOOST_CHECK(p3.has_hessians());
    BOOST_CHECK(p3.has_hessians_sparsity());

    BOOST_CHECK(!p4.has_gradient());
    BOOST_CHECK(!p4.has_gradient_sparsity());
    BOOST_CHECK(!p4.has_hessians());
    BOOST_CHECK(!p4.has_hessians_sparsity());

    problem p6{ss_not_impl{}};
    BOOST_CHECK_THROW(p6.set_seed(32u), not_implemented_error);

    problem p7{base_p{}};
    BOOST_CHECK_THROW(p7.set_seed(32u), not_implemented_error);

    // These two implement the has_sparsity() methods without the sparsity() methods.
    // They will not error out because the lack of the sparsity() methods makes the has_sparsity()
    // methods return always false.
    BOOST_CHECK_NO_THROW(problem{hgs_not_impl{}});
    BOOST_CHECK(!problem{hgs_not_impl{}}.has_gradient_sparsity());
    BOOST_CHECK_NO_THROW(problem{hgs_not_impl{}}.gradient_sparsity());
    BOOST_CHECK_NO_THROW(problem{hhs_not_impl{}});
    BOOST_CHECK(!problem{hhs_not_impl{}}.has_hessians_sparsity());
    BOOST_CHECK_NO_THROW(problem{hhs_not_impl{}}.hessians_sparsity());
}

BOOST_AUTO_TEST_CASE(problem_getters_test)
{
    vector_double lb_2(2, 13);
    vector_double ub_2(2, 17);
    vector_double fit_2(2, 1);
    vector_double grad_2{1, 1};
    sparsity_pattern grads_2_correct{{0, 0}, {0, 1}};
    std::vector<vector_double> hess_22{{1, 1}, {1, 1}};
    std::vector<sparsity_pattern> hesss_22_correct{{{0, 0}, {1, 0}}, {{0, 0}, {1, 0}}};

    problem p1{base_p(2, 3, 4, {3, 4, 5, 6, 7, 8, 9, 0, 1}, lb_2, ub_2)};
    problem p2{full_p(2, 0, 0, fit_2, lb_2, ub_2, grad_2, grads_2_correct, hess_22, hesss_22_correct)};
    problem p3{empty{}};

    BOOST_CHECK(p1.get_nobj() == 2);
    BOOST_CHECK(p1.get_nx() == 2);
    BOOST_CHECK(p1.get_nec() == 3);
    BOOST_CHECK(p1.get_nic() == 4);
    BOOST_CHECK(p1.get_nc() == 4 + 3);
    BOOST_CHECK((p1.get_c_tol() == vector_double{0., 0., 0., 0., 0., 0., 0.}));
    BOOST_CHECK(p1.get_nf() == 2 + 3 + 4);
    BOOST_CHECK((p1.get_bounds() == std::pair<vector_double, vector_double>{{13, 13}, {17, 17}}));

    // Making some evaluations
    auto N = 1235u;
    for (auto i = 0u; i < N; ++i) {
        p2.fitness({0, 0});
        p2.gradient({0, 0});
        p2.hessians({0, 0});
    }
    BOOST_CHECK(p2.get_fevals() == N);
    BOOST_CHECK(p2.get_gevals() == N);
    BOOST_CHECK(p2.get_hevals() == N);

    // User implemented
    BOOST_CHECK(p1.get_name() == "A base toy problem");
    BOOST_CHECK(p1.get_extra_info() == "Nothing to report");
    // Default
    BOOST_CHECK(p3.get_name() == typeid(*p3.extract<empty>()).name());
    BOOST_CHECK(p3.get_extra_info() == "");
}

BOOST_AUTO_TEST_CASE(problem_serialization_test)
{
    // Do the checking with the full problem.
    problem p{full_p{}}, p2{p};
    // Call objfun, grad and hess to increase the internal counters.
    p.fitness({1.});
    p.gradient({1.});
    p.hessians({1.});
    // Store the string representation.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(p);
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(p);
    }
    // Change the content of p before deserializing.
    p = problem{grad_p{}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(p);
    }
    auto after = boost::lexical_cast<std::string>(p);
    BOOST_CHECK_EQUAL(before, after);
    // Check that the properties of base_p where restored as well.
    BOOST_CHECK_EQUAL(p.extract<full_p>()->m_nobj, p2.extract<full_p>()->m_nobj);
    BOOST_CHECK_EQUAL(p.extract<full_p>()->m_nec, p2.extract<full_p>()->m_nec);
    BOOST_CHECK_EQUAL(p.extract<full_p>()->m_nic, p2.extract<full_p>()->m_nic);
    BOOST_CHECK(p.extract<full_p>()->m_ret_fit == p2.extract<full_p>()->m_ret_fit);
    BOOST_CHECK(p.extract<full_p>()->m_lb == p2.extract<full_p>()->m_lb);
    BOOST_CHECK(p.extract<full_p>()->m_ub == p2.extract<full_p>()->m_ub);
}

// Full minimal problems to test constraints number
// Only equality
struct c_01 {
    vector_double fitness(const vector_double &) const
    {
        return {2, 2};
    }
    vector_double::size_type get_nec() const
    {
        return 1u;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
};

// Only inequality
struct c_02 {
    vector_double fitness(const vector_double &) const
    {
        return {2, 2};
    }
    vector_double::size_type get_nic() const
    {
        return 1u;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
};

// Both equality and inequality
struct c_03 {
    vector_double fitness(const vector_double &) const
    {
        return {2, 2, 2};
    }
    vector_double::size_type get_nec() const
    {
        return 1u;
    }
    vector_double::size_type get_nic() const
    {
        return 1u;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
};
BOOST_AUTO_TEST_CASE(problem_constraint_dimension_test)
{
    BOOST_CHECK(problem{c_01{}}.get_nec() == 1u);
    BOOST_CHECK(problem{c_01{}}.get_nic() == 0u);
    BOOST_CHECK(problem{c_01{}}.get_nc() == 1u);
    BOOST_CHECK(problem{c_02{}}.get_nec() == 0u);
    BOOST_CHECK(problem{c_02{}}.get_nic() == 1u);
    BOOST_CHECK(problem{c_02{}}.get_nc() == 1u);
    BOOST_CHECK(problem{c_03{}}.get_nec() == 1u);
    BOOST_CHECK(problem{c_03{}}.get_nic() == 1u);
    BOOST_CHECK(problem{c_03{}}.get_nc() == 2u);
}

struct s_02 {
    vector_double fitness(const vector_double &) const
    {
        return {2, 2, 2};
    }
    vector_double::size_type get_nec() const
    {
        return 1u;
    }
    vector_double::size_type get_nic() const
    {
        return 1u;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
    void set_seed(unsigned int seed)
    {
        m_seed = seed;
    }
    unsigned int m_seed = 0u;
};

struct s_03 {
    vector_double fitness(const vector_double &) const
    {
        return {2, 2, 2};
    }
    vector_double::size_type get_nec() const
    {
        return 1u;
    }
    vector_double::size_type get_nic() const
    {
        return 1u;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
    void set_seed(unsigned int seed)
    {
        m_seed = seed;
    }
    bool has_set_seed() const
    {
        return false;
    }
    unsigned int m_seed = 0u;
};

BOOST_AUTO_TEST_CASE(problem_stochastic_test)
{
    problem prob{s_02{}};
    BOOST_CHECK(prob.is_stochastic() == true);
    BOOST_CHECK(prob.has_set_seed() == true);
    prob.set_seed(32u);
    BOOST_CHECK(prob.extract<s_02>()->m_seed == 32u);
    BOOST_CHECK(problem{s_03{}}.is_stochastic() == false);
    BOOST_CHECK(problem{s_03{}}.has_set_seed() == false);
}

struct extra_info_case {
    vector_double fitness(const vector_double &) const
    {
        return {2, 2, 2};
    }
    vector_double::size_type get_nec() const
    {
        return 1u;
    }
    vector_double::size_type get_nic() const
    {
        return 1u;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
    void set_seed(unsigned int seed)
    {
        m_seed = seed;
    }
    bool has_set_seed() const
    {
        return true;
    }
    std::string get_extra_info() const
    {
        return std::to_string(m_seed);
    }
    unsigned int m_seed = 0u;
};

BOOST_AUTO_TEST_CASE(problem_extra_info_test)
{
    problem prob{extra_info_case{}};
    problem prob2(prob);
    BOOST_CHECK(prob.get_extra_info() == prob2.get_extra_info());
    prob.set_seed(32u);
    BOOST_CHECK(prob.get_extra_info() == "32");
}

struct with_get_nobj {
    vector_double fitness(const vector_double &) const
    {
        return {2, 2, 2};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
    vector_double::size_type get_nobj() const
    {
        return 3u;
    }
};

struct without_get_nobj {
    vector_double fitness(const vector_double &) const
    {
        return {2, 2, 2};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
};

BOOST_AUTO_TEST_CASE(problem_get_nobj_detection)
{
    BOOST_CHECK(problem{with_get_nobj{}}.get_nobj() == 3u);
    BOOST_CHECK(problem{without_get_nobj{}}.get_nobj() == 1u);
    BOOST_CHECK_NO_THROW(problem{with_get_nobj{}}.fitness({1.}));
    BOOST_CHECK_THROW(problem{without_get_nobj{}}.fitness({1.}),
                      std::invalid_argument); // detects a returned size of 3 but has the default
}

BOOST_AUTO_TEST_CASE(problem_auto_sparsity_test)
{
    problem p{base_p(2u, 2u, 2u, {1., 1., 1., 1., 1., 1.}, {0., 0.}, {1., 1.})};
    BOOST_CHECK(p.gradient_sparsity() == detail::dense_gradient(6u, 2u));
    BOOST_CHECK(p.hessians_sparsity() == detail::dense_hessians(6u, 2u));
}

BOOST_AUTO_TEST_CASE(problem_get_set_c_tol_test)
{
    problem prob{base_p(2u, 1u, 1u, {1., 1., 1., 1.}, {0., 0.}, {1., 1.})};
    BOOST_CHECK((prob.get_c_tol() == vector_double{0., 0.}));
    prob.set_c_tol({1., 2.});
    BOOST_CHECK((prob.get_c_tol() == vector_double{1., 2.}));
    prob.set_c_tol({12., 22.});
    BOOST_CHECK((prob.get_c_tol() == vector_double{12., 22.}));
    if (std::numeric_limits<double>::has_quiet_NaN) {
        BOOST_CHECK_THROW(prob.set_c_tol({std::numeric_limits<double>::quiet_NaN(), 22.}), std::invalid_argument);
        BOOST_CHECK((prob.get_c_tol() == vector_double{12., 22.}));
    }
    BOOST_CHECK_THROW(prob.set_c_tol({-12., 22.}), std::invalid_argument);
    BOOST_CHECK((prob.get_c_tol() == vector_double{12., 22.}));
    BOOST_CHECK_THROW(prob.set_c_tol({12., 22., 33.});, std::invalid_argument);
    BOOST_CHECK((prob.get_c_tol() == vector_double{12., 22.}));

    // checking the overload method
    BOOST_CHECK_THROW(prob.set_c_tol(-12.), std::invalid_argument);
    if (std::numeric_limits<double>::has_quiet_NaN) {
        BOOST_CHECK_THROW(prob.set_c_tol(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    }
    prob.set_c_tol(22);
    BOOST_CHECK((prob.get_c_tol() == vector_double{22., 22.}));
}

BOOST_AUTO_TEST_CASE(problem_feasibility_methods_test)
{
    problem test01{base_p(2u, 1u, 1u, {1., 1., 1., 1.}, {0., 0.}, {1., 1.})};
    problem test02{base_p(2u, 1u, 1u, {1., 1., 1e-9, -1.}, {0., 0.}, {1., 1.})};

    BOOST_CHECK(test01.feasibility_x({1., 1.}) == false);
    BOOST_CHECK(test01.feasibility_f({2., 3., 1e-10, 3.}) == false);
    test01.set_c_tol({2., 2.});
    BOOST_CHECK(test01.feasibility_x({1., 1.}) == true);
    BOOST_CHECK(test01.feasibility_f({2., 3., 1e-10, 1.}) == true);

    BOOST_CHECK(test02.feasibility_x({1., 1.}) == false);
    BOOST_CHECK(test02.feasibility_f({2., 3., 1e-10, 3.}) == false);
    test02.set_c_tol({2., 2.});
    BOOST_CHECK(test02.feasibility_x({1., 1.}) == true);
    BOOST_CHECK(test02.feasibility_f({2., 3., 1e-10, 1.5}) == true);

    BOOST_CHECK_THROW(test02.feasibility_f({1., -23, 1e-10, 2., 34.}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(null_problem_test)
{
    // Problem instantiation
    problem p{null_problem{}};
    BOOST_CHECK_EQUAL(p.get_name(), "Null problem");
    // Pick a few reference points
    vector_double x1 = {1};
    vector_double x2 = {2};
    // Fitness test
    BOOST_CHECK((p.fitness(x1) == vector_double{0}));
    BOOST_CHECK((p.fitness(x2) == vector_double{0}));
    p = problem{null_problem{2}};
    BOOST_CHECK(null_problem{2}.get_nobj() == 2u);
    BOOST_CHECK(null_problem{2}.get_nec() == 0u);
    BOOST_CHECK(null_problem{2}.get_nic() == 0u);
    BOOST_CHECK(null_problem{2}.get_nix() == 0u);
    BOOST_CHECK((null_problem{2, 3, 4, 1}.get_nobj() == 2u));
    BOOST_CHECK((null_problem{2, 3, 4, 1}.get_nec() == 3u));
    BOOST_CHECK((null_problem{2, 3, 4, 1}.get_nic() == 4u));
    BOOST_CHECK((null_problem{2, 3, 4, 1}.get_nix() == 1u));
    BOOST_CHECK((null_problem{2, 3, 4, 0}.get_nix() == 0u));
    BOOST_CHECK(p.get_nobj() == 2u);
    BOOST_CHECK((p.fitness(x1) == vector_double{0, 0}));
    BOOST_CHECK((p.fitness(x2) == vector_double{0, 0}));
    BOOST_CHECK_THROW(p = problem{null_problem{0}}, std::invalid_argument);
    BOOST_CHECK_THROW((p = problem{null_problem{2, 3, 4, 2}}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(null_problem_serialization_test)
{
    problem p{null_problem{2, 3, 4}};
    // Call objfun to increase the internal counter.
    p.fitness({1});
    // Store the string representation of p.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(p);
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(p);
    }
    // Change the content of p before deserializing.
    p = problem{null_problem{}};
    BOOST_CHECK_EQUAL(p.get_nobj(), 1u);
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(p);
    }
    auto after = boost::lexical_cast<std::string>(p);
    BOOST_CHECK_EQUAL(before, after);
    BOOST_CHECK_EQUAL(p.get_nobj(), 2u);
    BOOST_CHECK_EQUAL(p.get_nec(), 3u);
    BOOST_CHECK_EQUAL(p.get_nic(), 4u);
    BOOST_CHECK_EQUAL(p.fitness({1.}).size(), 9u);
}

BOOST_AUTO_TEST_CASE(extract_test)
{
    problem p{null_problem{}};
    BOOST_CHECK(p.is<null_problem>());
    BOOST_CHECK(!p.is<base_p>());
    BOOST_CHECK((std::is_same<null_problem *, decltype(p.extract<null_problem>())>::value));
    BOOST_CHECK(
        (std::is_same<null_problem const *, decltype(static_cast<const problem &>(p).extract<null_problem>())>::value));
    BOOST_CHECK(p.extract<null_problem>() != nullptr);
    BOOST_CHECK(static_cast<const problem &>(p).extract<null_problem>() != nullptr);
    BOOST_CHECK(p.extract<base_p>() == nullptr);
    BOOST_CHECK(static_cast<const problem &>(p).extract<base_p>() == nullptr);
}

struct ts1 {
    vector_double fitness(const vector_double &) const
    {
        return {2, 2, 2};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
};

struct ts2 {
    vector_double fitness(const vector_double &) const
    {
        return {2, 2, 2};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
    thread_safety get_thread_safety() const
    {
        return thread_safety::none;
    }
};

struct ts3 {
    vector_double fitness(const vector_double &) const
    {
        return {2, 2, 2};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
    int get_thread_safety() const
    {
        return 2;
    }
};

BOOST_AUTO_TEST_CASE(thread_safety_test)
{
    BOOST_CHECK(problem{null_problem{}}.get_thread_safety() == thread_safety::basic);
    BOOST_CHECK(problem{ts1{}}.get_thread_safety() == thread_safety::basic);
    BOOST_CHECK(problem{ts2{}}.get_thread_safety() == thread_safety::none);
    BOOST_CHECK(problem{ts3{}}.get_thread_safety() == thread_safety::basic);
}

struct gs1 {
    vector_double fitness(const vector_double &) const
    {
        return {0, 0};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}};
    }
    sparsity_pattern gradient_sparsity() const
    {
        if (!n_grad_invs) {
            ++n_grad_invs;
            return {};
        }
        return {{0, 0}};
    }
    static int n_grad_invs;
};

int gs1::n_grad_invs = 0;

struct gs2 {
    vector_double fitness(const vector_double &) const
    {
        return {0, 0};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}};
    }
    sparsity_pattern gradient_sparsity() const
    {
        return {{0, 0}};
    }
};

struct gs3 {
    vector_double fitness(const vector_double &) const
    {
        return {0, 0};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}};
    }
    sparsity_pattern gradient_sparsity() const
    {
        return {{0, 0}, {0, 2}, {0, 1}};
    }
};

BOOST_AUTO_TEST_CASE(custom_gs)
{
    // Test a gradient sparsity that changes after the first invocation of gradient_sparsity().
    problem p{gs1{}};
    BOOST_CHECK_THROW(p.gradient_sparsity(), std::invalid_argument);
    p = problem{gs2{}};
    BOOST_CHECK_NO_THROW(p.gradient_sparsity());
    // Gradient sparsity not sorted.
    BOOST_CHECK_THROW(p = problem{gs3{}}, std::invalid_argument);
}

struct hs1 {
    vector_double fitness(const vector_double &) const
    {
        return {0, 0};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}};
    }
    vector_double::size_type get_nobj() const
    {
        return 2;
    }
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        if (!n_hess_invs) {
            ++n_hess_invs;
            return {{{1, 0}}, {{1, 0}}};
        }
        return {{{1, 0}}, {{1, 0}, {2, 0}}};
    }
    static int n_hess_invs;
};

int hs1::n_hess_invs = 0;

struct hs2 {
    vector_double fitness(const vector_double &) const
    {
        return {0, 0};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}};
    }
    vector_double::size_type get_nobj() const
    {
        return 2;
    }
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return {{{1, 0}}, {{1, 0}, {2, 0}}};
    }
};

struct hs3 {
    vector_double fitness(const vector_double &) const
    {
        return {0, 0};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}};
    }
    vector_double::size_type get_nobj() const
    {
        return 2;
    }
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return {{{1, 0}, {2, 1}, {1, 1}}, {{1, 0}, {2, 0}}};
    }
};

BOOST_AUTO_TEST_CASE(custom_hs)
{
    // Test a hessians sparsity that changes after the first invocation of hessians_sparsity().
    problem p{hs1{}};
    BOOST_CHECK_THROW(p.hessians_sparsity(), std::invalid_argument);
    p = problem{hs2{}};
    BOOST_CHECK_NO_THROW(p.hessians_sparsity());
    BOOST_CHECK_THROW(p = problem{hs3{}}, std::invalid_argument);
}

struct hess1 {
    vector_double fitness(const vector_double &) const
    {
        return {0, 0};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}};
    }
    vector_double::size_type get_nobj() const
    {
        return 2;
    }
    std::vector<vector_double> hessians(const vector_double &) const
    {
        return {{}};
    }
};

BOOST_AUTO_TEST_CASE(broken_hessian)
{
    // Test a hessians method that returns a number of vectors different from get_nf().
    problem p{hess1{}};
    BOOST_CHECK_THROW(p.hessians({1, 1, 1, 1, 1, 1}), std::invalid_argument);
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
        return 1u;
    }
    vector_double::size_type get_nec() const
    {
        return 1u;
    }
    vector_double::size_type get_nic() const
    {
        return 1u;
    }
    vector_double::size_type get_nix() const
    {
        return m_nix;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{1, 1, 1}, {2, 2, 2}};
    }
    std::string get_name() const
    {
        return "A minlp problem";
    }
    vector_double::size_type m_nix;
};

BOOST_AUTO_TEST_CASE(minlp_test)
{
    BOOST_CHECK((problem{minlp{1u}}.get_nix() == 1u));
    BOOST_CHECK((problem{minlp{1u}}.get_ncx() == 2u));
    BOOST_CHECK((problem{minlp{1u}}.get_nx() == 3u));
    BOOST_CHECK((problem{minlp{2u}}.get_nix() == 2u));
    BOOST_CHECK((problem{minlp{2u}}.get_ncx() == 1u));
    BOOST_CHECK((problem{minlp{2u}}.get_nx() == 3u));
    BOOST_CHECK((problem{minlp{3u}}.get_nix() == 3u));
    BOOST_CHECK((problem{minlp{3u}}.get_ncx() == 0u));
    BOOST_CHECK((problem{minlp{3u}}.get_nx() == 3u));
    BOOST_CHECK_THROW(problem{minlp{5u}}, std::invalid_argument);
}