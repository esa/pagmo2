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

#ifndef PAGMO_IPOPT_HPP
#define PAGMO_IPOPT_HPP

#include <pagmo/config.hpp>

#if defined(PAGMO_WITH_IPOPT)

#include <IpIpoptCalculatedQuantities.hpp>
#include <IpIpoptData.hpp>
#include <IpTNLP.hpp>
#include <algorithm>
#include <boost/numeric/conversion/cast.hpp>
#include <cassert>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

struct ipopt_nlp final : Ipopt::TNLP {
    // Some shortcuts from the Ipopt namespace.
    using Index = Ipopt::Index;
    using Number = Ipopt::Number;
    static_assert(std::is_same<Number, double>::value, "I hate everybody.");
    using SolverReturn = Ipopt::SolverReturn;
    using IpoptData = Ipopt::IpoptData;
    using IpoptCalculatedQuantities = Ipopt::IpoptCalculatedQuantities;

    // Ctor from problem.
    ipopt_nlp(const problem &prob) : m_prob(prob)
    {
        // Check the problem is single-objective.
        if (m_prob.get_nobj() > 1u) {
            pagmo_throw(std::invalid_argument,
                        std::to_string(m_prob.get_nobj()) + " objectives were detected in the input problem named '"
                            + m_prob.get_name()
                            + "', but the ipopt algorithm can solve only single-objective problems");
        }

        // Conversion of the sparsity information to the format required by Ipopt.
        // Gradients first.
        {
            // NOTE: our format for the gradient sparsity matches almost exactly Ipopt's. The only difference
            // is that we also report the sparsity for the objective function's gradient, while Ipopt's jacobian
            // contains only constraints' gradients. Thus, we will need to discard the the objfun's sparsity
            // information and to decrease by one the row indices in the pattern (i.e., a first index of zero in
            // a pattern element must refer to the first constraint).
            // https://www.coin-or.org/Ipopt/documentation/node22.html
            const auto sp = prob.gradient_sparsity();
            // Determine where the gradients of the constraints start.
            const auto it = std::lower_bound(sp.begin(), sp.end(), sparsity_pattern::value_type(1u, 0u));
            // Transform it into the Ipopt format.
            std::transform(it, sp.end(), std::back_inserter(m_jac_sp), [](const sparsity_pattern::value_type &p) {
                return std::make_pair(boost::numeric_cast<Index>(p.first - 1u), boost::numeric_cast<Index>(p.second));
            });
        }

        // Hessians.
        {
            // NOTE: Ipopt requires a single sparsity pattern for the hessian of the lagrangian (that is,
            // the pattern must be valid for objfun and all constraints), but we provide a separate sparsity pattern for
            // objfun and every constraint. We will thus need to merge our sparsity patterns in a single sparsity
            // pattern.
            // https://www.coin-or.org/Ipopt/documentation/node22.html
            const auto sps = m_prob.hessians_sparsity();
            sparsity_pattern merged_sp;
            if (m_prob.has_hessians_sparsity()) {
                for (const auto &sp : sps) {
                    // NOTE: we need to create a separate copy each time as std::merge() requires distinct ranges.
                    const auto old_merged_sp(merged_sp);
                    merged_sp.clear();
                    std::merge(old_merged_sp.begin(), old_merged_sp.end(), sp.begin(), sp.end(),
                               std::back_inserter(merged_sp));
                }
            } else {
                // If the hessians sparsity is not user-provided, we don't need the merge operation:
                // all patterns are identical, just pick the first one.
                merged_sp = sps[0];
            }
            // Convert into Index pairs.
            std::transform(merged_sp.begin(), merged_sp.end(), std::back_inserter(m_lag_sp),
                           [](const sparsity_pattern::value_type &p) {
                               return std::make_pair(boost::numeric_cast<Index>(p.first),
                                                     boost::numeric_cast<Index>(p.second));
                           });
        }
    }

    // Default dtor.
    ~ipopt_nlp() = default;

    // Delete everything else.
    ipopt_nlp(const ipopt_nlp &) = delete;
    ipopt_nlp(ipopt_nlp &&) = delete;
    ipopt_nlp &operator=(const ipopt_nlp &) = delete;
    ipopt_nlp &operator=(ipopt_nlp &&) = delete;

    // Method to return some info about the nlp.
    virtual bool get_nlp_info(Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag,
                              IndexStyleEnum &index_style) override final
    {
        // Number of dimensions of the problem.
        n = boost::numeric_cast<Index>(m_prob.get_nx());

        // Total number of constraints.
        m = boost::numeric_cast<Index>(m_prob.get_nc());

        // Number of nonzero entries in the jacobian.
        nnz_jac_g = boost::numeric_cast<Index>(m_jac_sp.size());

        // Number of nonzero entries in the hessian of the lagrangian.
        nnz_h_lag = boost::numeric_cast<Index>(m_lag_sp.size());

        // We use C style indexing (0-based).
        index_style = TNLP::C_STYLE;

        return true;
    }

    // Method to return the bounds of the problem.
    virtual bool get_bounds_info(Index n, Number *x_l, Number *x_u, Index m, Number *g_l, Number *g_u) override final
    {
        assert(n == boost::numeric_cast<Index>(m_prob.get_nx()));
        assert(m == boost::numeric_cast<Index>(m_prob.get_nc()));
        (void)n;

        // Box bounds.
        const auto bounds = m_prob.get_bounds();
        // Lower bounds.
        std::copy(bounds.first.begin(), bounds.first.end(), x_l);
        // Upper bounds.
        std::copy(bounds.second.begin(), bounds.second.end(), x_u);

        // Equality constraints: lb == ub == 0.
        std::fill(g_l, g_l + m_prob.get_nec(), 0.);
        std::fill(g_u, g_u + m_prob.get_nec(), 0.);

        // Inequality constraints: lb == -inf, ub == 0.
        std::fill(g_l + m_prob.get_nec(), g_l + m, std::numeric_limits<double>::has_infinity
                                                       ? -std::numeric_limits<double>::infinity()
                                                       : std::numeric_limits<double>::lowest());
        std::fill(g_u + m_prob.get_nec(), g_u + m, 0.);

        return true;
    }

    // Method to return the starting point for the algorithm.
    virtual bool get_starting_point(Index n, bool init_x, Number *x, bool init_z, Number *z_L, Number *z_U, Index m,
                                    bool init_lambda, Number *lambda) override final
    {
    }

    // Method to return the objective value.
    virtual bool eval_f(Index n, const Number *x, bool new_x, Number &obj_value) override final
    {
    }

    // Method to return the gradient of the objective.
    virtual bool eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f) override final
    {
    }

    // Method to return the constraint residuals.
    virtual bool eval_g(Index n, const Number *x, bool new_x, Index m, Number *g) override final
    {
    }

    // Method to return:
    // 1) The structure of the jacobian (if "values" is NULL)
    // 2) The values of the jacobian (if "values" is not NULL)
    virtual bool eval_jac_g(Index n, const Number *x, bool new_x, Index m, Index nele_jac, Index *iRow, Index *jCol,
                            Number *values) override final
    {
    }

    // Method to return:
    // 1) The structure of the hessian of the lagrangian (if "values" is NULL)
    // 2) The values of the hessian of the lagrangian (if "values" is not NULL)
    virtual bool eval_h(Index n, const Number *x, bool new_x, Number obj_factor, Index m, const Number *lambda,
                        bool new_lambda, Index nele_hess, Index *iRow, Index *jCol, Number *values) override final
    {
    }

    // Solution Methods.
    // This method is called when the algorithm is complete so the TNLP can store/write the solution.
    virtual void finalize_solution(SolverReturn status, Index n, const Number *x, const Number *z_L, const Number *z_U,
                                   Index m, const Number *g, const Number *lambda, Number obj_value,
                                   const IpoptData *ip_data, IpoptCalculatedQuantities *ip_cq) override final
    {
    }

    // Data members.
    const problem &m_prob;
    // Jacobian sparsity pattern as required by Ipopt: sparse
    // rectangular matrix represented as a list of (Row,Col)
    // pairs.
    // https://www.coin-or.org/Ipopt/documentation/node38.html
    std::vector<std::pair<Index, Index>> m_jac_sp;
    // Same format for the hessian of the lagrangian (but it's a square matrix).
    std::vector<std::pair<Index, Index>> m_lag_sp;
};
}

class ipopt
{
public:
    population evolve(population pop) const
    {
    }
    template <typename Archive>
    void save(Archive &ar) const
    {
    }
    template <typename Archive>
    void load(Archive &ar)
    {
    }
};
}

PAGMO_REGISTER_ALGORITHM(pagmo::ipopt)

#else // PAGMO_WITH_IPOPT

#error The ipopt.hpp header was included, but pagmo was not compiled with Ipopt support

#endif // PAGMO_WITH_IPOPT

#endif
