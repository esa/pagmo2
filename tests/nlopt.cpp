/* Copyright 2017 PaGMO development team

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

#define BOOST_TEST_MODULE nlopt_test
#include <boost/test/included/unit_test.hpp>

#include <boost/any.hpp>
#include <cmath>
#include <nlopt.h>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/rosenbrock.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(nlopt_construction)
{
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
    auto b{a};
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
    auto tmp{*a.extract<nlopt>()};
    auto d{std::move(tmp)};
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