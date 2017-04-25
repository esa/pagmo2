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

#include <IpIpoptApplication.hpp>
#include <IpIpoptCalculatedQuantities.hpp>
#include <IpIpoptData.hpp>
#include <IpSmartPtr.hpp>
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
    static_assert(std::is_same<Number, double>::value, "");
    using SolverReturn = Ipopt::SolverReturn;
    using IpoptData = Ipopt::IpoptData;
    using IpoptCalculatedQuantities = Ipopt::IpoptCalculatedQuantities;

    // Ctor from problem.
    ipopt_nlp(const problem &prob, vector_double start) : m_prob(prob), m_start(std::move(start))
    {
        assert(m_start.size() == prob.get_nx());

        // Check the problem is single-objective.
        if (m_prob.get_nobj() > 1u) {
            pagmo_throw(std::invalid_argument,
                        std::to_string(m_prob.get_nobj()) + " objectives were detected in the input problem named '"
                            + m_prob.get_name()
                            + "', but the ipopt algorithm can solve only single-objective problems");
        }

        // We need the gradient.
        if (!m_prob.has_gradient()) {
            pagmo_throw(std::invalid_argument, "the ipopt algorithm needs the gradient, but the problem named '"
                                                   + m_prob.get_name() + "' does not provide it");
        }

        // Prepare the dv used for fitness computation.
        m_dv.resize(m_start.size());

        // This will contain the solution.
        m_sol.resize(m_start.size());

        // Final values of the constraints.
        m_final_eq.resize(m_prob.get_nec());
        m_final_ineq.resize(m_prob.get_nic());

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
            // Transform it into the Ipopt format: convert to Index and decrease the first index of each pair by one.
            std::transform(it, sp.end(), std::back_inserter(m_jac_sp), [](const sparsity_pattern::value_type &p) {
                return std::make_pair(boost::numeric_cast<Index>(p.first - 1u), boost::numeric_cast<Index>(p.second));
            });
            if (m_prob.has_gradient_sparsity()) {
                // Store the objfun gradient sparsity, if user-provided.
                std::copy(sp.begin(), it, std::back_inserter(m_obj_g_sp));
            }
        }

        // Hessians.
        {
            // NOTE: Ipopt requires a single sparsity pattern for the hessian of the lagrangian (that is,
            // the pattern must be valid for objfun and all constraints), but we provide a separate sparsity pattern for
            // objfun and every constraint. We will thus need to merge our sparsity patterns in a single sparsity
            // pattern.
            // https://www.coin-or.org/Ipopt/documentation/node22.html
            sparsity_pattern merged_sp;
            if (m_prob.has_hessians_sparsity()) {
                // Store the original hessians sparsity only if it is user-provided.
                m_h_sp = m_prob.hessians_sparsity();
                for (const auto &sp : m_h_sp) {
                    // NOTE: we need to create a separate copy each time as std::set_union() requires distinct ranges.
                    const auto old_merged_sp(merged_sp);
                    merged_sp.clear();
                    std::set_union(old_merged_sp.begin(), old_merged_sp.end(), sp.begin(), sp.end(),
                                   std::back_inserter(merged_sp));
                }
            } else {
                // If the hessians sparsity is not user-provided, we don't need the merge operation:
                // all patterns are dense and identical. Like this, we avoid using a huge amount of memory
                // by calling hessians_sparsity().
                merged_sp = detail::dense_hessian(m_prob.get_nx());
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
    virtual bool get_starting_point(Index n, bool init_x, Number *x, bool init_z, Number *, Number *, Index m,
                                    bool init_lambda, Number *) override final
    {
        // NOTE: these come from the tutorial. I think the values of these asserts depend on
        // the configuration of the Ipopt run.
        assert(init_x);
        assert(!init_z);
        assert(!init_lambda);
        assert(n == boost::numeric_cast<Index>(m_prob.get_nx()));
        assert(n == boost::numeric_cast<Index>(m_start.size()));
        assert(m == boost::numeric_cast<Index>(m_prob.get_nc()));
        (void)init_x;
        (void)init_z;
        (void)init_lambda;
        (void)n;
        (void)m;

        std::copy(m_start.begin(), m_start.end(), x);

        return true;
    }

    // Method to return the objective value.
    virtual bool eval_f(Index n, const Number *x, bool new_x, Number &obj_value) override final
    {
        assert(n == boost::numeric_cast<Index>(m_prob.get_nx()));
        // NOTE: the new_x boolean flag will be false if the last call to any of the eval_* function
        // used the same x value. Probably we can ignore this in favour of the upcoming caches work.
        (void)new_x;

        std::copy(x, x + n, m_dv.begin());
        obj_value = m_prob.fitness(m_dv)[0];

        return true;
    }

    // Method to return the gradient of the objective.
    virtual bool eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f) override final
    {
        assert(n == boost::numeric_cast<Index>(m_prob.get_nx()));
        (void)new_x;

        std::copy(x, x + n, m_dv.begin());
        // Compute the full gradient (this includes the cosntraints as well).
        const auto gradient = m_prob.gradient(m_dv);

        if (m_prob.has_gradient_sparsity()) {
            // Sparse gradient case.
            auto g_it = gradient.begin();

            // First we fill the dense output gradient with zeroes.
            std::fill(grad_f, grad_f + n, 0.);
            // Then we iterate over the sparsity pattern of the objfun, and fill in the
            // nonzero bits in grad_f.
            for (auto it = m_obj_g_sp.begin(); it != m_obj_g_sp.end(); ++it, ++g_it) {
                assert(it->first == 0u);
                assert(g_it != gradient.end());
                grad_f[it->second] = *g_it;
            }
        } else {
            // Dense gradient.
            std::copy(gradient.data(), gradient.data() + n, grad_f);
        }

        return true;
    }

    // Value of the constraints.
    virtual bool eval_g(Index n, const Number *x, bool new_x, Index m, Number *g) override final
    {
        assert(n == boost::numeric_cast<Index>(m_prob.get_nx()));
        assert(m == boost::numeric_cast<Index>(m_prob.get_nc()));
        (void)new_x;

        std::copy(x, x + n, m_dv.begin());
        const auto fitness = m_prob.fitness(m_dv);

        // Eq. constraints.
        std::copy(fitness.data() + 1, fitness.data() + 1 + m_prob.get_nec(), g);

        // Ineq. constraints.
        std::copy(fitness.data() + 1 + m_prob.get_nec(), fitness.data() + 1 + m, g + m_prob.get_nec());

        return true;
    }

    // Method to return:
    // 1) The structure of the jacobian (if "values" is NULL)
    // 2) The values of the jacobian (if "values" is not NULL)
    virtual bool eval_jac_g(Index n, const Number *x, bool new_x, Index m, Index nele_jac, Index *iRow, Index *jCol,
                            Number *values) override final
    {
        assert(n == boost::numeric_cast<Index>(m_prob.get_nx()));
        assert(m == boost::numeric_cast<Index>(m_prob.get_nc()));
        assert(nele_jac == boost::numeric_cast<Index>(m_jac_sp.size()));
        (void)new_x;

        if (values) {
            std::copy(x, x + n, m_dv.begin());
            const auto gradient = m_prob.gradient(m_dv);
            // NOTE: here we need the gradients of the constraints only, so we need to discard the gradient of the
            // objfun. If the gradient sparsity is user-provided, then the size of the objfun sparse gradient is
            // m_obj_g_sp.size(), otherwise the gradient is dense and its size is nx.
            std::copy(gradient.data() + (m_prob.has_gradient_sparsity() ? m_obj_g_sp.size() : m_prob.get_nx()),
                      gradient.data() + gradient.size(), values);
        } else {
            for (decltype(m_jac_sp.size()) k = 0; k < m_jac_sp.size(); ++k) {
                iRow[k] = m_jac_sp[k].first;
                jCol[k] = m_jac_sp[k].second;
            }
        }

        return true;
    }

    // Method to return:
    // 1) The structure of the hessian of the lagrangian (if "values" is NULL)
    // 2) The values of the hessian of the lagrangian (if "values" is not NULL)
    virtual bool eval_h(Index n, const Number *x, bool new_x, Number obj_factor, Index m, const Number *lambda,
                        bool new_lambda, Index nele_hess, Index *iRow, Index *jCol, Number *values) override final
    {
        assert(n == boost::numeric_cast<Index>(m_prob.get_nx()));
        assert(m == boost::numeric_cast<Index>(m_prob.get_nc()));
        assert(nele_hess == boost::numeric_cast<Index>(m_lag_sp.size()));
        (void)new_x;
        (void)new_lambda;

        if (!m_prob.has_hessians()) {
            // If the problem does not provide hessians, return false. Ipopt will
            // do a numerical estimation.
            return false;
        }

        if (values) {
            std::copy(x, x + n, m_dv.begin());
            const auto hessians = m_prob.hessians(m_dv);
            if (m_prob.has_hessians_sparsity()) {
                // Sparse case.
                // Objfun first.
                assert(hessians[0].size() <= m_lag_sp.size());
                auto it_h_sp = m_h_sp[0].begin();
                auto it = hessians[0].begin();
                assert(hessians[0].size() == m_h_sp[0].size());
                // NOTE: the idea here is that we need to fill up values with m_lag_sp.size()
                // numbers. Some of these numbers will be zero because, in general, our hessians
                // may contain fewer elements. In order to establish which elements to take
                // from our hessians and which elements to set to zero, we need to iterate at the
                // same time on the original sparsity pattern and compare the indices pairs.
                for (decltype(m_lag_sp.size()) i = 0; i < m_lag_sp.size(); ++i) {
                    // NOTE: static_cast is ok as we already converted via numeric_cast
                    // earlier.
                    if (it_h_sp != m_h_sp[0].end() && static_cast<Index>(it_h_sp->first) == m_lag_sp[i].first
                        && static_cast<Index>(it_h_sp->second) == m_lag_sp[i].second) {
                        assert(it != hessians[0].end());
                        values[i] = (*it) * obj_factor;
                        ++it;
                        ++it_h_sp;
                    } else {
                        values[i] = 0.;
                    }
                }
                // Constraints.
                for (decltype(hessians.size()) j = 1; j < hessians.size(); ++j) {
                    assert(hessians[j].size() <= m_lag_sp.size());
                    it_h_sp = m_h_sp[j].begin();
                    it = hessians[j].begin();
                    assert(hessians[j].size() == m_h_sp[j].size());
                    const auto lam = lambda[j - 1u];
                    for (decltype(m_lag_sp.size()) i = 0; i < m_lag_sp.size(); ++i) {
                        if (it_h_sp != m_h_sp[j].end() && static_cast<Index>(it_h_sp->first) == m_lag_sp[i].first
                            && static_cast<Index>(it_h_sp->second) == m_lag_sp[i].second) {
                            assert(it != hessians[j].end());
                            values[i] += (*it) * lam;
                            ++it;
                            ++it_h_sp;
                        }
                    }
                }
            } else {
                // Dense case.
                // First the objfun.
                assert(hessians[0].size() == m_lag_sp.size());
                std::transform(hessians[0].begin(), hessians[0].end(), values,
                               [obj_factor](double a) { return obj_factor * a; });
                // The constraints (to be added iteratively to the existing values).
                for (decltype(hessians.size()) i = 1; i < hessians.size(); ++i) {
                    assert(hessians[i].size() == m_lag_sp.size());
                    // NOTE: the lambda factors refer to the constraints only, hence we need
                    // to decrease i by 1.
                    const auto lam = lambda[i - 1u];
                    std::transform(hessians[i].begin(), hessians[i].end(), values, values,
                                   [lam](double a, double b) { return b + lam * a; });
                }
            }
        } else {
            // Fill in the sp of the hessian of the lagrangian.
            for (decltype(m_lag_sp.size()) k = 0; k < m_lag_sp.size(); ++k) {
                iRow[k] = m_lag_sp[k].first;
                jCol[k] = m_lag_sp[k].second;
            }
        }

        return true;
    }

    // Solution Methods.
    // This method is called when the algorithm is complete so the TNLP can store/write the solution.
    virtual void finalize_solution(SolverReturn status, Index n, const Number *x, const Number *, const Number *,
                                   Index m, const Number *g, const Number *, Number obj_value, const IpoptData *,
                                   IpoptCalculatedQuantities *) override final
    {
        assert(n == boost::numeric_cast<Index>(m_prob.get_nx()));
        assert(m == boost::numeric_cast<Index>(m_prob.get_nc()));

        // Store the solution.
        std::copy(x, x + n, m_sol.begin());

        // Store the values of the constraints.
        std::copy(g, g + m_prob.get_nec(), m_final_eq.begin());
        std::copy(g + m_prob.get_nec(), g + m, m_final_ineq.begin());

        // Store the final value of the objfun.
        m_final_objfun = obj_value;

        // Store the status.
        m_status = status;
    }

    // Data members.
    // The pagmo problem.
    const problem &m_prob;
    // Initial guess.
    const vector_double m_start;
    // Temporary dv used for fitness computation.
    vector_double m_dv;
    // Dv of the solution.
    vector_double m_sol;
    // Final values of the constraints.
    vector_double m_final_eq;
    vector_double m_final_ineq;
    // Final value of the objfun.
    double m_final_objfun;
    // Status at the end of the optimisation.
    SolverReturn m_status;
    // Sparsity pattern of the gradient of the objfun. We need this for the evaluation
    // of the gradient in eval_grad_f(). If the gradient sparsity is not user-provided,
    // it will be empty.
    sparsity_pattern m_obj_g_sp;
    // The original hessians sp from pagmo. We need this if the hessians sparsity
    // is user-provided, as we must rebuild the hessian of the lagrangian in Ipopt format.
    // If the hessians sparsity is not user-provided, it will be empty.
    std::vector<sparsity_pattern> m_h_sp;
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
        auto initial_guess = pop.get_x()[pop.best_idx()];
        Ipopt::SmartPtr<Ipopt::TNLP> nlp = ::new detail::ipopt_nlp(pop.get_problem(), initial_guess);
        Ipopt::SmartPtr<Ipopt::IpoptApplication> app = ::IpoptApplicationFactory();
        app->Options()->SetNumericValue("tol", 1e-9);
        app->Options()->SetStringValue("hessian_approximation", "limited-memory");

        Ipopt::ApplicationReturnStatus status = app->Initialize();
        if (status != Ipopt::Solve_Succeeded) {
            throw;
        }
        status = app->OptimizeTNLP(nlp);
        if (status != Ipopt::Solve_Succeeded) {
            throw;
        }
        pop.set_x(pop.best_idx(), dynamic_cast<detail::ipopt_nlp &>(*nlp).m_sol);
        return pop;
    }
    std::string get_name() const
    {
        return "Ipopt";
    }
    template <typename Archive>
    void save(Archive &) const
    {
    }
    template <typename Archive>
    void load(Archive &)
    {
    }
};
}

PAGMO_REGISTER_ALGORITHM(pagmo::ipopt)

#else // PAGMO_WITH_IPOPT

#error The ipopt.hpp header was included, but pagmo was not compiled with Ipopt support

#endif // PAGMO_WITH_IPOPT

#endif
