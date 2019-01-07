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

#define BOOST_TEST_MODULE generic_utilities_test
#include <boost/test/included/unit_test.hpp>

#include <limits>
#include <stdexcept>

#include <pagmo/io.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/gradients_and_hessians.hpp>

using namespace pagmo;

struct dummy_problem {
    vector_double fitness(const vector_double &dv) const
    {
        vector_double retval(3, 0.);
        retval[0] = dv[0] * dv[0] - std::sin(dv[1] - dv[2]) + dv[3]; // dense
        retval[1] = dv[1] - dv[2] + dv[3];
        retval[2] = std::exp(dv[2]);
        return retval;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        vector_double lb(4, -15);
        vector_double ub(4, 15);
        return {lb, ub};
    }
    vector_double::size_type get_nobj() const
    {
        return 3u;
    }
};

// returning a different fitness dimension according to mood :)
struct dummy_problem_malformed {
    vector_double fitness(const vector_double &dv) const
    {
        vector_double retval;
        retval.push_back(dv[0] * dv[0] - std::sin(dv[1] - dv[2]) + dv[3]); // dense
        retval.push_back(dv[1] - dv[0] + dv[3]);
        if (dv[0] == 0.1) {
            retval.push_back(std::exp(dv[0]));
        }
        return retval;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        vector_double lb(4, -15);
        vector_double ub(4, 15);
        return {lb, ub};
    }
    vector_double::size_type get_nobj() const
    {
        return 3u;
    }
};

BOOST_AUTO_TEST_CASE(estimate_sparsity_test)
{
    {
        dummy_problem udp{};
        dummy_problem_malformed udp2{};
        auto sp
            = estimate_sparsity([udp](const vector_double &x) { return udp.fitness(x); }, {0.1, 0.2, 0.3, 0.4}, 1e-8);
        BOOST_CHECK((sp == sparsity_pattern{{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1}, {1, 2}, {1, 3}, {2, 2}}));
        BOOST_CHECK_THROW(
            estimate_sparsity([udp2](const vector_double &x) { return udp2.fitness(x); }, {0.1, 0.2, 0.3, 0.4}, 1e-8),
            std::invalid_argument);
    }
    {
        problem prob{dummy_problem{}};
        problem prob2{dummy_problem_malformed{}};
        auto sp
            = estimate_sparsity([prob](const vector_double &x) { return prob.fitness(x); }, {0.1, 0.2, 0.3, 0.4}, 1e-8);
        BOOST_CHECK((sp == sparsity_pattern{{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1}, {1, 2}, {1, 3}, {2, 2}}));
        BOOST_CHECK_THROW(
            estimate_sparsity([prob2](const vector_double &x) { return prob2.fitness(x); }, {0.1, 0.2, 0.3, 0.4}, 1e-8),
            std::invalid_argument);
    }
}

struct dummy_problem_easy_grad {
    vector_double fitness(const vector_double &dv) const
    {
        vector_double retval(3, 0.);
        retval[0] = dv[0] + dv[1] * dv[1] + dv[2] * dv[2] + dv[3] * dv[3] * dv[3];
        retval[1] = dv[1] * dv[2] * dv[3] * dv[0];
        retval[2] = std::sin(dv[0]);
        return retval;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        vector_double lb(4, -15);
        vector_double ub(4, 15);
        return {lb, ub};
    }
    vector_double::size_type get_nobj() const
    {
        return 3u;
    }
};

BOOST_AUTO_TEST_CASE(estimate_gradient_test)
{

    dummy_problem_easy_grad udp{};
    dummy_problem_malformed udp2{};
    BOOST_CHECK_THROW(
        estimate_gradient([udp2](const vector_double &x) { return udp2.fitness(x); }, {0.1, 0.2, 0.3, 0.4}, 1e-8),
        std::invalid_argument);
    BOOST_CHECK_THROW(
        estimate_gradient_h([udp2](const vector_double &x) { return udp2.fitness(x); }, {0.1, 0.2, 0.3, 0.4}, 1e-8),
        std::invalid_argument);
    auto g = estimate_gradient([udp](const vector_double &x) { return udp.fitness(x); }, {0.1, 0.2, 0.3, 0.4}, 1e-8);
    auto gh = estimate_gradient_h([udp](const vector_double &x) { return udp.fitness(x); }, {0.1, 0.2, 0.3, 0.4}, 1e-2);
    vector_double res = {1, 0.4, 0.6, 0.48, 0.024, 0.012, 0.008, 0.006, 0.9950041652780257660956, 0, 0, 0};
    for (unsigned i = 0u; i < res.size(); ++i) {
        BOOST_CHECK_CLOSE(gh[i], res[i], 1e-7);
    }
    for (unsigned i = 0u; i < res.size(); ++i) {
        BOOST_CHECK_CLOSE(gh[i], res[i], 1e-11);
    }
}