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

#include <pagmo/algorithm.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

class ipopt_nlp final : public Ipopt::TNLP
{
public:
    // Some shortcuts from the Ipopt namespace.
    using Index = Ipopt::Index;
    using Number = Ipopt::Number;
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

        // Store the gradient sparsity, but only if user-provided. We will optimise the dense
        // case and avoid stroring the pattern explicitly.
        if (m_prob.has_gradient_sparsity()) {
            m_g_sp = prob.gradient_sparsity();
        }

        // Same for the hessians.
        if (m_prob.has_hessians_sparsity()) {
            m_h_sp = prob.hessians_sparsity();
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
        // NOTE: our format for the gradient sparsity matches almost exactly Ipopt's. The only difference
        // is that we also report the sparsity for the objective function's gradient, while Ipopt is interested
        // only in the sparsity of the constraints. Thus we will need to discard the the objfun's sparsity
        // information.
        if (m_prob.has_gradient_sparsity()) {
            // Determine where the gradients of the constraints start.
            const auto it = std::lower_bound(m_g_sp.cbegin(), m_g_sp.cend(), sparsity_pattern::value_type(1u, 0u));

            // Need to do a bit of horrid overflow checking :/.
            using diff_type = std::iterator_traits<sparsity_pattern::const_iterator>::difference_type;
            using udiff_type = std::make_unsigned<diff_type>::type;
            if (m_g_sp.size() > static_cast<udiff_type>(std::numeric_limits<diff_type>::max())) {
                pagmo_throw(std::overflow_error, "Overflow error, the sparsity pattern size is too large.");
            }

            // nnz_jac_g is the distance between the first sparsity index referring to the constraints
            // and the end of the sparsity pattern.
            nnz_jac_g = boost::numeric_cast<Index>(std::distance(it, m_g_sp.cend()));
        } else {
            // Dense case.
            nnz_jac_g = boost::numeric_cast<Index>(m_prob.get_nx() * m_prob.get_nc());
        }

        // Number of nonzero entries in the constraints hessians.
        // NOTE: Ipopt requires a single sparsity pattern valid for all constraint hessians, but we
        // provide a separate sparsity pattern for every constraint. We will thus need to merge
        // our sparsity patterns in a single sparsity pattern. Also, we need to discard the sparsity
        // pattern of the hessiand of the objfun as Ipopt does not need that.
        if (m_prob.has_hessians_sparsity()) {
            sparsity_pattern merged_sp;
            for (auto it = m_h_sp.begin() + 1; it != m_h_sp.end(); ++it) {
                // NOTE: we need to create a separate copy each time as std::merge() requires distinct ranges.
                const auto old_merged_sp(merged_sp);
                merged_sp.clear();
                std::merge(old_merged_sp.begin(), old_merged_sp.end(), it->begin(), it->end(),
                           std::back_inserter(merged_sp));
            }
            nnz_h_lag = boost::numeric_cast<Index>(merged_sp.size());
        } else {
            // Dense case.
            // NOTE: this is the number of elements in the lower triangular half of a square matrix of nx * nx
            // (including the diagonal).
            nnz_h_lag = boost::numeric_cast<Index>(m_prob.get_nx() * (m_prob.get_nx() - 1u) / 2u + m_prob.get_nx());
        }

        // Use the C style indexing (0-based).
        index_style = TNLP::C_STYLE;
    }

    // Method to return the bounds of the problem.
    virtual bool get_bounds_info(Index n, Number *x_l, Number *x_u, Index m, Number *g_l, Number *g_u) override final
    {
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

private:
    const problem &m_prob;
    // Gradient sp.
    sparsity_pattern m_g_sp;
    // Hessians sp.
    std::vector<sparsity_pattern> m_h_sp;
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
