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

#define BOOST_TEST_MODULE ipopt_test
#include <boost/test/included/unit_test.hpp>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <vector>

#include <pagmo/algorithms/ipopt.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(ipopt_nlp_test)
{
    using ipopt_nlp = detail::ipopt_nlp;
    using Index = ipopt_nlp::Index;
    problem prob(hock_schittkowsky_71{});
    ipopt_nlp nlp(prob, {1.1, 1.2, 1.3, 1.4});

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
        (c_lb == vector_double{0., std::numeric_limits<double>::has_infinity ? -std::numeric_limits<double>::infinity()
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
}
