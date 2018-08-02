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

#ifndef PAGMO_IPOPT_HPP
#define PAGMO_IPOPT_HPP

#include <pagmo/config.hpp>

#if defined(PAGMO_WITH_IPOPT)

#include <IpIpoptApplication.hpp>
#include <IpIpoptCalculatedQuantities.hpp>
#include <IpIpoptData.hpp>
#include <IpReturnCodes.hpp>
#include <IpSmartPtr.hpp>
#include <IpTNLP.hpp>
#include <algorithm>
#include <boost/any.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>

namespace pagmo
{

namespace detail
{

// Usual trick with global read-only data useful to the Ipopt wrapper.
template <typename = void>
struct ipopt_data {
    // A map to link a human-readable description to Ipopt return codes.
    // NOTE: in C++11 hashing of enums might not be available. Provide our own.
    struct res_hasher {
        std::size_t operator()(Ipopt::ApplicationReturnStatus res) const noexcept
        {
            return std::hash<int>{}(static_cast<int>(res));
        }
    };
    using result_map_t = std::unordered_map<Ipopt::ApplicationReturnStatus, std::string, res_hasher>;
    static const result_map_t results;
};

#define PAGMO_DETAIL_IPOPT_RES_ENTRY(name)                                                                             \
    {                                                                                                                  \
        Ipopt::name, #name " (value = " + std::to_string(static_cast<int>(Ipopt::name)) + ")"                          \
    }

// Static init.
template <typename T>
const typename ipopt_data<T>::result_map_t ipopt_data<T>::results
    = {PAGMO_DETAIL_IPOPT_RES_ENTRY(Solve_Succeeded),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Solved_To_Acceptable_Level),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Infeasible_Problem_Detected),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Search_Direction_Becomes_Too_Small),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Diverging_Iterates),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(User_Requested_Stop),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Feasible_Point_Found),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Maximum_Iterations_Exceeded),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Restoration_Failed),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Error_In_Step_Computation),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Not_Enough_Degrees_Of_Freedom),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Invalid_Problem_Definition),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Invalid_Option),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Invalid_Number_Detected),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Unrecoverable_Exception),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(NonIpopt_Exception_Thrown),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Insufficient_Memory),
       PAGMO_DETAIL_IPOPT_RES_ENTRY(Internal_Error)};

#undef PAGMO_DETAIL_IPOPT_RES_ENTRY

// The NLP implementation required by Ipopt's C++ interface.
struct ipopt_nlp final : Ipopt::TNLP {
    // Single entry of the log (objevals, objval, n of unsatisfied const, constr. violation, feasibility).
    using log_line_type = std::tuple<unsigned long, double, vector_double::size_type, double, bool>;
    // The log.
    using log_type = std::vector<log_line_type>;

    // Some shortcuts from the Ipopt namespace.
    using Index = Ipopt::Index;
    using Number = Ipopt::Number;
    static_assert(std::is_same<Number, double>::value, "");
    using SolverReturn = Ipopt::SolverReturn;
    using IpoptData = Ipopt::IpoptData;
    using IpoptCalculatedQuantities = Ipopt::IpoptCalculatedQuantities;

    // Ctor from problem, initial guess, verbosity.
    ipopt_nlp(const problem &prob, vector_double start, unsigned verbosity)
        : m_prob(prob), m_start(std::move(start)), m_verbosity(verbosity)
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
        // NOTE: these try catches and the mechanism to handle exceptions outside the Ipopt
        // callbacks are needed because, apparently, Ipopt does not handle gracefully exceptions
        // thrown from the callbacks (I suspect this has something to do with the support of
        // interfaces for languages other than C++?). This is the same approach we adopt in the
        // NLopt wrapper: trap everything in a try/catch block, and store the exception for re-throw
        // in ipopt::evolve(). In case of errors we return "false" from the callback, as this
        // signals to the the Ipopt API that something went wrong.
        try {
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
            // LCOV_EXCL_START
        } catch (...) {
            m_eptr = std::current_exception();
            return false;
            // LCOV_EXCL_STOP
        }
    }

    // Method to return the bounds of the problem.
    virtual bool get_bounds_info(Index n, Number *x_l, Number *x_u, Index m, Number *g_l, Number *g_u) override final
    {
        try {
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
            std::fill(g_l + m_prob.get_nec(), g_l + m,
                      std::numeric_limits<double>::has_infinity ? -std::numeric_limits<double>::infinity()
                                                                : std::numeric_limits<double>::lowest());
            std::fill(g_u + m_prob.get_nec(), g_u + m, 0.);

            return true;
            // LCOV_EXCL_START
        } catch (...) {
            m_eptr = std::current_exception();
            return false;
            // LCOV_EXCL_STOP
        }
    }

    // Method to return the starting point for the algorithm.
    virtual bool get_starting_point(Index n, bool init_x, Number *x, bool init_z, Number *, Number *, Index m,
                                    bool init_lambda, Number *) override final
    {
        try {
            assert(n == boost::numeric_cast<Index>(m_prob.get_nx()));
            assert(n == boost::numeric_cast<Index>(m_start.size()));
            assert(m == boost::numeric_cast<Index>(m_prob.get_nc()));
            (void)n;
            (void)m;

            if (init_x) {
                std::copy(m_start.begin(), m_start.end(), x);
            }

            // LCOV_EXCL_START
            if (init_z) {
                pagmo_throw(std::runtime_error,
                            "we are being asked to provide initial values for the bounds multiplier by "
                            "the Ipopt API, but in pagmo we do not support them");
            }

            if (init_lambda) {
                pagmo_throw(std::runtime_error,
                            "we are being asked to provide initial values for the constraints multiplier by "
                            "the Ipopt API, but in pagmo we do not support them");
            }

            return true;
        } catch (...) {
            m_eptr = std::current_exception();
            return false;
            // LCOV_EXCL_STOP
        }
    }

    // Method to return the objective value.
    virtual bool eval_f(Index n, const Number *x, bool new_x, Number &obj_value) override final
    {
        try {
            assert(n == boost::numeric_cast<Index>(m_prob.get_nx()));
            // NOTE: the new_x boolean flag will be false if the last call to any of the eval_* function
            // used the same x value. Probably we can ignore this in favour of the upcoming caches work.
            (void)new_x;

            std::copy(x, x + n, m_dv.begin());
            const auto fitness = m_prob.fitness(m_dv);
            obj_value = fitness[0];

            // Update the log if requested.
            if (m_verbosity && !(m_objfun_counter % m_verbosity)) {
                // Constraints bits.
                const auto ctol = m_prob.get_c_tol();
                const auto c1eq = detail::test_eq_constraints(fitness.data() + 1, fitness.data() + 1 + m_prob.get_nec(),
                                                              ctol.data());
                const auto c1ineq
                    = detail::test_ineq_constraints(fitness.data() + 1 + m_prob.get_nec(),
                                                    fitness.data() + fitness.size(), ctol.data() + m_prob.get_nec());
                // This will be the total number of violated constraints.
                const auto nv = m_prob.get_nc() - c1eq.first - c1ineq.first;
                // This will be the norm of the violation.
                const auto l = c1eq.second + c1ineq.second;
                // Test feasibility.
                const auto feas = m_prob.feasibility_f(fitness);

                if (!(m_objfun_counter / m_verbosity % 50u)) {
                    // Every 50 lines print the column names.
                    print("\n", std::setw(10), "objevals:", std::setw(15), "objval:", std::setw(15),
                          "violated:", std::setw(15), "viol. norm:", '\n');
                }
                // Print to screen the log line.
                print(std::setw(10), m_objfun_counter + 1u, std::setw(15), obj_value, std::setw(15), nv, std::setw(15),
                      l, feas ? "" : " i", '\n');
                // Record the log.
                m_log.emplace_back(m_objfun_counter + 1u, obj_value, nv, l, feas);
            }

            // Update the counter.
            ++m_objfun_counter;

            return true;
        } catch (...) {
            m_eptr = std::current_exception();
            return false;
        }
    }

    // Method to return the gradient of the objective.
    virtual bool eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f) override final
    {
        try {
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
        } catch (...) {
            m_eptr = std::current_exception();
            return false;
        }
    }

    // Value of the constraints.
    virtual bool eval_g(Index n, const Number *x, bool new_x, Index m, Number *g) override final
    {
        try {
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
        } catch (...) {
            m_eptr = std::current_exception();
            return false;
        }
    }

    // Method to return:
    // 1) The structure of the jacobian (if "values" is NULL)
    // 2) The values of the jacobian (if "values" is not NULL)
    virtual bool eval_jac_g(Index n, const Number *x, bool new_x, Index m, Index nele_jac, Index *iRow, Index *jCol,
                            Number *values) override final
    {
        try {
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
        } catch (...) {
            m_eptr = std::current_exception();
            return false;
        }
    }

    // Method to return:
    // 1) The structure of the hessian of the lagrangian (if "values" is NULL)
    // 2) The values of the hessian of the lagrangian (if "values" is not NULL)
    virtual bool eval_h(Index n, const Number *x, bool new_x, Number obj_factor, Index m, const Number *lambda,
                        bool new_lambda, Index nele_hess, Index *iRow, Index *jCol, Number *values) override final
    {
        try {
            assert(n == boost::numeric_cast<Index>(m_prob.get_nx()));
            assert(m == boost::numeric_cast<Index>(m_prob.get_nc()));
            assert(nele_hess == boost::numeric_cast<Index>(m_lag_sp.size()));
            (void)new_x;
            (void)new_lambda;

            if (!m_prob.has_hessians()) {
                pagmo_throw(
                    std::invalid_argument,
                    "the exact evaluation of the Hessian of the Lagrangian was requested, but the problem named '"
                        + m_prob.get_name()
                        + "' does not provide it. Please consider providing the Hessian or, alternatively, "
                          "set the option 'hessian_approximation' to 'limited-memory' in the ipopt algorithm options");
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
                            // This means that we are at a sparsity entry which is both in our original sparsity
                            // pattern and in the merged one.
                            assert(it != hessians[0].end());
                            values[i] = (*it) * obj_factor;
                            ++it;
                            ++it_h_sp;
                        } else {
                            // This means we are at a sparsity entry which is in the merged patterns but not in our
                            // original sparsity pattern. Thus, set the value to zero.
                            values[i] = 0.;
                        }
                    }
                    // Constraints.
                    for (decltype(hessians.size()) j = 1; j < hessians.size(); ++j) {
                        assert(hessians[j].size() <= m_lag_sp.size());
                        it_h_sp = m_h_sp[j].begin();
                        it = hessians[j].begin();
                        assert(hessians[j].size() == m_h_sp[j].size());
                        // NOTE: the lambda factors refer to the constraints only, hence we need
                        // to decrease i by 1.
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
        } catch (...) {
            m_eptr = std::current_exception();
            return false;
        }
    }

    // Solution Methods.
    // This method is called when the algorithm is complete so the TNLP can store/write the solution.
    // NOTE: no need for try/catch here, nothing can throw.
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
    // Verbosity.
    const unsigned m_verbosity;
    // Objfun counter.
    unsigned long m_objfun_counter = 0;
    // Log.
    log_type m_log;
    // This exception pointer will be null, unless an error is raised in one of the virtual methods. If not null, it
    // will be re-thrown in the ipopt::evolve() method.
    std::exception_ptr m_eptr;
};
} // namespace detail

/// Ipopt.
/**
 * \image html ipopt.png "COIN_OR logo." width=3cm
 *
 * \verbatim embed:rst:leading-asterisk
 * .. versionadded:: 2.2
 * \endverbatim
 *
 * This class is a user-defined algorithm (UDA) that wraps the Ipopt (Interior Point OPTimizer) solver,
 * a software package for large-scale nonlinear optimization. Ipopt is a powerful solver that
 * is able to handle robustly and efficiently constrained nonlinear opimization problems at high dimensionalities.
 *
 * Ipopt supports only single-objective minimisation, and it requires the availability of the gradient in the
 * optimisation problem. If possible, for best results the Hessians should be provided as well (but Ipopt
 * can estimate numerically the Hessians if needed).
 *
 * In order to support pagmo's population-based optimisation model, ipopt::evolve() will select
 * a single individual from the input pagmo::population to be optimised.
 * If the optimisation produces a better individual (as established by pagmo::compare_fc()),
 * the optimised individual will be inserted back into the population.
 * The selection and replacement strategies can be configured via set_selection(const std::string &),
 * set_selection(population::size_type), set_replacement(const std::string &) and
 * set_replacement(population::size_type).
 *
 * Configuring the optimsation run
 * -------------------------------
 *
 * Ipopt supports a large amount of options for the configuration of the optimisation run. The options
 * are divided into three categories:
 * - *string* options (i.e., the type of the option is ``std::string``),
 * - *integer* options (i.e., the type of the option is ``Ipopt::Index`` - an alias for some integer type, typically
 *   ``int``),
 * - *numeric* options (i.e., the type of the option is ``double``).
 *
 * The full list of options is available on the
 * <a href="https://www.coin-or.org/Ipopt/documentation/node40.html">Ipopt website</a>. pagmo::ipopt allows to configure
 * any Ipopt option via methods such as ipopt::set_string_options(), ipopt::set_string_option(),
 * ipopt::set_integer_options(), etc., which need to be used before invoking ipopt::evolve().
 *
 * If the user does not set any option, pagmo::ipopt will use Ipopt's default values for the options (see the
 * <a href="https://www.coin-or.org/Ipopt/documentation/node40.html">documentation</a>), with the following
 * modifications:
 * - if the ``"print_level"`` integer option is **not** set by the user, it will be set to 0 by pagmo::ipopt (this will
 *   suppress most screen output produced by the solver - note that we support an alternative form of logging via
 *   the ipopt::set_verbosity() method);
 * - if the ``"hessian_approximation"`` string option is **not** set by the user and the optimisation problem does
 *   **not** provide the Hessians, then the option will be set to ``"limited-memory"`` by pagmo::ipopt. This makes it
 *   possible to optimise problems without Hessians out-of-the-box (i.e., Ipopt will approximate numerically the
 *   Hessians for you);
 * - if the ``"constr_viol_tol"`` numeric option is **not** set by the user and the optimisation problem is constrained,
 *   then pagmo::ipopt will compute the minimum value ``min_tol`` in the vector returned by pagmo::problem::get_c_tol()
 *   for the optimisation problem at hand. If ``min_tol`` is nonzero, then the ``"constr_viol_tol"`` Ipopt option will
 *   be set to ``min_tol``, otherwise the default Ipopt value (1E-4) will be used for the option. This ensures that,
 *   if the constraint tolerance is not explicitly set by the user, a solution deemed feasible by Ipopt is also
 *   deemed feasible by pagmo (but the opposite is not necessarily true).
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    A moved-from :cpp:class:`pagmo::ipopt` is destructible and assignable. Any other operation will result
 *    in undefined behaviour.
 *
 * .. note::
 *
 *    This user-defined algorithm is available only if pagmo was compiled with the ``PAGMO_WITH_IPOPT`` option
 *    enabled (see the :ref:`installation instructions <install>`).
 *
 * .. seealso::
 *
 *    https://projects.coin-or.org/Ipopt.
 *
 * \endverbatim
 */
class ipopt : public not_population_based
{
    template <typename Pair>
    static void opt_checker(bool status, const Pair &p, const std::string &op_type)
    {
        if (!status) {
            pagmo_throw(std::invalid_argument, "failed to set the ipopt " + op_type + " option '" + p.first
                                                   + "' to the value: " + detail::to_string(p.second));
        }
    }

public:
    /// Single data line for the algorithm's log.
    /**
     * A log data line is a tuple consisting of:
     * - the number of objective function evaluations made so far,
     * - the objective function value for the current decision vector,
     * - the number of constraints violated by the current decision vector,
     * - the constraints violation norm for the current decision vector,
     * - a boolean flag signalling the feasibility of the current decision vector.
     */
    using log_line_type = std::tuple<unsigned long, double, vector_double::size_type, double, bool>;
    /// Log type.
    /**
     * The algorithm log is a collection of ipopt::log_line_type data lines, stored in chronological order
     * during the optimisation if the verbosity of the algorithm is set to a nonzero value
     * (see ipopt::set_verbosity()).
     */
    using log_type = std::vector<log_line_type>;

private:
    static_assert(std::is_same<log_line_type, detail::ipopt_nlp::log_line_type>::value, "Invalid log line type.");

public:
    /// Evolve population.
    /**
     * This method will select an individual from \p pop, optimise it with Ipopt, replace an individual in \p pop with
     * the optimised individual, and finally return \p pop.
     * The individual selection and replacement criteria can be set via set_selection(const std::string &),
     * set_selection(population::size_type), set_replacement(const std::string &) and
     * set_replacement(population::size_type). The return status of the Ipopt optimisation run will be recorded (it can
     * be fetched with get_last_opt_result()).
     *
     * @param pop the population to be optimised.
     *
     * @return the optimised population.
     *
     * @throws std::invalid_argument in the following cases:
     * - the population's problem is multi-objective or it does not provide the gradient,
     * - the setup of the Ipopt solver options fails (e.g., an invalid option was specified by the user),
     * - the components of the individual selected for optimisation contain NaNs or they are outside
     *   the problem's bounds,
     * - the exact evaluation of the Hessians was requested, but the problem does not support it.
     * @throws std::runtime_error if the initialization of the Ipopt solver fails.
     * @throws unspecified any exception thrown by the public interface of pagmo::problem or
     * pagmo::not_population_based.
     */
    population evolve(population pop) const
    {
        if (!pop.size()) {
            // In case of an empty pop, just return it.
            return pop;
        }

        auto &prob = pop.get_problem();

        // Setup of the initial guess. Store also the original fitness
        // of the selected individual, old_f, for later use.
        auto sel_xf = select_individual(pop);
        vector_double initial_guess(std::move(sel_xf.first)), old_f(std::move(sel_xf.second));

        // Check the initial guess.
        // NOTE: this should be guaranteed by the population's invariants.
        assert(initial_guess.size() == prob.get_nx());
        const auto bounds = prob.get_bounds();
        for (decltype(bounds.first.size()) i = 0; i < bounds.first.size(); ++i) {
            if (std::isnan(initial_guess[i])) {
                pagmo_throw(std::invalid_argument,
                            "the value of the initial guess at index " + std::to_string(i) + " is NaN");
            }
            if (initial_guess[i] < bounds.first[i] || initial_guess[i] > bounds.second[i]) {
                pagmo_throw(std::invalid_argument, "the value of the initial guess at index " + std::to_string(i)
                                                       + " is outside the problem's bounds");
            }
        }

        // Initialize the Ipopt machinery, following the tutorial.
        Ipopt::SmartPtr<Ipopt::TNLP> nlp = ::new detail::ipopt_nlp(pop.get_problem(), initial_guess, m_verbosity);
        // Store a reference to the derived class for later use.
        detail::ipopt_nlp &inlp = dynamic_cast<detail::ipopt_nlp &>(*nlp);
        Ipopt::SmartPtr<Ipopt::IpoptApplication> app = ::IpoptApplicationFactory();
        app->RethrowNonIpoptException(true);

        // Logic for the handling of constraints tolerances. The logic is as follows:
        // - if the user provides the "constr_viol_tol" option, use that *unconditionally*. Otherwise,
        // - compute the minimum tolerance min_tol among those provided by the problem. If zero, ignore
        //   it and use the ipopt default value for "constr_viol_tol" (1e-4). Otherwise, use min_tol as the value for
        //   "constr_viol_tol".
        if (prob.get_nc() && !m_numeric_opts.count("constr_viol_tol")) {
            const auto c_tol = prob.get_c_tol();
            assert(!c_tol.empty());
            const double min_tol = *std::min_element(c_tol.begin(), c_tol.end());
            if (min_tol > 0.) {
                const auto tmp_p = std::make_pair(std::string("constr_viol_tol"), min_tol);
                opt_checker(app->Options()->SetNumericValue(tmp_p.first, tmp_p.second), tmp_p, "numeric");
            }
        }

        // Logic for the hessians computation:
        // - if the problem does *not* provide the hessians, and the "hessian_approximation" is *not*
        //   set, then we set it to "limited-memory".
        // This way, problems without hessians will work out of the box.
        if (!prob.has_hessians() && !m_string_opts.count("hessian_approximation")) {
            const auto tmp_p = std::make_pair(std::string("hessian_approximation"), std::string("limited-memory"));
            opt_checker(app->Options()->SetStringValue(tmp_p.first, tmp_p.second), tmp_p, "string");
        }

        // Logic for print_level: change the default to zero.
        if (!m_integer_opts.count("print_level")) {
            const auto tmp_p = std::make_pair(std::string("print_level"), Ipopt::Index(0));
            opt_checker(app->Options()->SetIntegerValue(tmp_p.first, tmp_p.second), tmp_p, "integer");
        }

        // Set the other options.
        for (const auto &p : m_string_opts) {
            opt_checker(app->Options()->SetStringValue(p.first, p.second), p, "string");
        }
        for (const auto &p : m_numeric_opts) {
            opt_checker(app->Options()->SetNumericValue(p.first, p.second), p, "numeric");
        }
        for (const auto &p : m_integer_opts) {
            opt_checker(app->Options()->SetIntegerValue(p.first, p.second), p, "integer");
        }

        // NOTE: Initialize() can take a filename as input, defaults to "ipopt.opt". This is a file
        // which is supposed to contain ipopt's options. Since we can set the options from the code,
        // let's disable this functionality by passing an empty string.
        const Ipopt::ApplicationReturnStatus status = app->Initialize("");
        if (status != Ipopt::Solve_Succeeded) {
            // LCOV_EXCL_START
            pagmo_throw(std::runtime_error,
                        "the initialisation of the ipopt algorithm failed. The return status code is: "
                            + detail::ipopt_data<>::results.at(status));
            // LCOV_EXCL_STOP
        }
        // Run the optimisation.
        m_last_opt_res = app->OptimizeTNLP(nlp);
        if (m_verbosity) {
            // Print to screen the result of the optimisation, if we are being verbose.
            std::cout << "\nOptimisation return status: " << detail::ipopt_data<>::results.at(m_last_opt_res) << '\n';
        }
        // Replace the log.
        m_log = std::move(inlp.m_log);

        // Handle any exception that might've been thrown.
        if (inlp.m_eptr) {
            std::rethrow_exception(inlp.m_eptr);
        }

        // Compute the new fitness vector.
        const auto new_f = prob.fitness(inlp.m_sol);

        // Store the new individual into the population, but only if better.
        if (compare_fc(new_f, old_f, prob.get_nec(), prob.get_c_tol())) {
            replace_individual(pop, inlp.m_sol, new_f);
        }

        // Return the evolved pop.
        return pop;
    }
    /// Get the result of the last optimisation.
    /**
     * @return the result of the last evolve() call, or ``Ipopt::Solve_Succeeded`` if no optimisations have been
     * run yet.
     */
    Ipopt::ApplicationReturnStatus get_last_opt_result() const
    {
        return m_last_opt_res;
    }
    /// Get the algorithm's name.
    /**
     * @return <tt>"Ipopt"</tt>.
     */
    std::string get_name() const
    {
        return "Ipopt: Interior Point Optimization";
    }
    /// Get extra information about the algorithm.
    /**
     * @return a human-readable string containing useful information about the algorithm's properties
     * (e.g., the Ipopt optimisation options, the selection/replacement policies, etc.).
     */
    std::string get_extra_info() const
    {
        return "\tLast optimisation return code: " + detail::ipopt_data<>::results.at(m_last_opt_res)
               + "\n\tVerbosity: " + std::to_string(m_verbosity) + "\n\tIndividual selection "
               + (boost::any_cast<population::size_type>(&m_select)
                      ? "idx: " + std::to_string(boost::any_cast<population::size_type>(m_select))
                      : "policy: " + boost::any_cast<std::string>(m_select))
               + "\n\tIndividual replacement "
               + (boost::any_cast<population::size_type>(&m_replace)
                      ? "idx: " + std::to_string(boost::any_cast<population::size_type>(m_replace))
                      : "policy: " + boost::any_cast<std::string>(m_replace))
               + (m_string_opts.size() ? "\n\tString options: " + detail::to_string(m_string_opts) : "")
               + (m_integer_opts.size() ? "\n\tInteger options: " + detail::to_string(m_integer_opts) : "")
               + (m_numeric_opts.size() ? "\n\tNumeric options: " + detail::to_string(m_numeric_opts) : "") + "\n";
    }
    /// Set verbosity.
    /**
     * This method will set the algorithm's verbosity. If \p n is zero, no output is produced during the optimisation
     * and no logging is performed. If \p n is nonzero, then every \p n objective function evaluations the status
     * of the optimisation will be both printed to screen and recorded internally. See ipopt::log_line_type and
     * ipopt::log_type for information on the logging format. The internal log can be fetched via get_log().
     *
     * Example (verbosity 1):
     * @code{.unparsed}
     * objevals:        objval:      violated:    viol. norm:
     *         1        48.9451              1        1.25272 i
     *         2         30.153              1       0.716591 i
     *         3        26.2884              1        1.04269 i
     *         4        14.6958              2        7.80753 i
     *         5        14.7742              2        5.41342 i
     *         6         17.093              1      0.0905025 i
     *         7        17.1772              1      0.0158448 i
     *         8        17.0254              2      0.0261289 i
     *         9        17.0162              2     0.00435195 i
     *        10        17.0142              2    0.000188461 i
     *        11         17.014              1    1.90997e-07 i
     *        12         17.014              0              0
     * @endcode
     * The ``i`` at the end of some rows indicates that the decision vector is infeasible. Feasibility
     * is checked against the problem's tolerance.
     *
     * By default, the verbosity level is zero.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    The number of constraints violated, the constraints violation norm and the feasibility flag stored in the log
     *    are all determined via the facilities and the tolerances specified within :cpp:class:`pagmo::problem`. That
     *    is, they might not necessarily be consistent with Ipopt's notion of feasibility. See the explanation
     *    of how the ``"constr_viol_tol"`` numeric option is handled in :cpp:class:`pagmo::ipopt`.
     *
     * .. note::
     *
     *    Ipopt supports its own logging format and protocol, including the ability to print to screen and write to
     *    file. Ipopt's screen logging is disabled by default (i.e., the Ipopt verbosity setting is set to 0 - see
     *    :cpp:class:`pagmo::ipopt`). On-screen logging can be enabled via the ``"print_level"`` string option.
     *
     * \endverbatim
     *
     * @param n the desired verbosity level.
     */
    void set_verbosity(unsigned n)
    {
        m_verbosity = n;
    }
    /// Get the optimisation log.
    /**
     * See ipopt::log_type for a description of the optimisation log. Logging is turned on/off via
     * set_verbosity().
     *
     * @return a const reference to the log.
     */
    const log_type &get_log() const
    {
        return m_log;
    }
    /// Save to archive.
    /**
     * @param ar the target archive.
     *
     * @throws unspecified any exception thrown by the serialization of primitive types or pagmo::not_population_based.
     */
    template <typename Archive>
    void save(Archive &ar) const
    {
        ar(cereal::base_class<not_population_based>(this), m_string_opts, m_integer_opts, m_numeric_opts,
           m_last_opt_res, m_verbosity, m_log);
    }
    /// Load from archive.
    /**
     * In case of exceptions, \p this will be reset to a default-constructed state.
     *
     * @param ar the source archive.
     *
     * @throws unspecified any exception thrown by the deserialization of primitive types or
     * pagmo::not_population_based.
     */
    template <typename Archive>
    void load(Archive &ar)
    {
        try {
            ar(cereal::base_class<not_population_based>(this), m_string_opts, m_integer_opts, m_numeric_opts,
               m_last_opt_res, m_verbosity, m_log);
            // LCOV_EXCL_START
        } catch (...) {
            *this = ipopt{};
            throw;
            // LCOV_EXCL_STOP
        }
    }
    /// Set string option.
    /**
     * This method will set the optimisation string option \p name to \p value.
     * The optimisation options are passed to the Ipopt API when calling evolve().
     *
     * @param name of the option.
     * @param value of the option.
     */
    void set_string_option(const std::string &name, const std::string &value)
    {
        m_string_opts[name] = value;
    }
    /// Set integer option.
    /**
     * This method will set the optimisation integer option \p name to \p value.
     * The optimisation options are passed to the Ipopt API when calling evolve().
     *
     * @param name of the option.
     * @param value of the option.
     */
    void set_integer_option(const std::string &name, Ipopt::Index value)
    {
        m_integer_opts[name] = value;
    }
    /// Set numeric option.
    /**
     * This method will set the optimisation numeric option \p name to \p value.
     * The optimisation options are passed to the Ipopt API when calling evolve().
     *
     * @param name of the option.
     * @param value of the option.
     */
    void set_numeric_option(const std::string &name, double value)
    {
        m_numeric_opts[name] = value;
    }
    /// Set string options.
    /**
     * This method will set the optimisation string options contained in \p m.
     * It is equivalent to calling set_string_option() passing all the name-value pairs in \p m
     * as arguments.
     *
     * @param m the name-value map that will be used to set the options.
     */
    void set_string_options(const std::map<std::string, std::string> &m)
    {
        for (const auto &p : m) {
            set_string_option(p.first, p.second);
        }
    }
    /// Set integer options.
    /**
     * This method will set the optimisation integer options contained in \p m.
     * It is equivalent to calling set_integer_option() passing all the name-value pairs in \p m
     * as arguments.
     *
     * @param m the name-value map that will be used to set the options.
     */
    void set_integer_options(const std::map<std::string, Ipopt::Index> &m)
    {
        for (const auto &p : m) {
            set_integer_option(p.first, p.second);
        }
    }
    /// Set numeric options.
    /**
     * This method will set the optimisation numeric options contained in \p m.
     * It is equivalent to calling set_numeric_option() passing all the name-value pairs in \p m
     * as arguments.
     *
     * @param m the name-value map that will be used to set the options.
     */
    void set_numeric_options(const std::map<std::string, double> &m)
    {
        for (const auto &p : m) {
            set_numeric_option(p.first, p.second);
        }
    }
    /// Get string options.
    /**
     * @return the name-value map of optimisation string options.
     */
    std::map<std::string, std::string> get_string_options() const
    {
        return m_string_opts;
    }
    /// Get integer options.
    /**
     * @return the name-value map of optimisation integer options.
     */
    std::map<std::string, Ipopt::Index> get_integer_options() const
    {
        return m_integer_opts;
    }
    /// Get numeric options.
    /**
     * @return the name-value map of optimisation numeric options.
     */
    std::map<std::string, double> get_numeric_options() const
    {
        return m_numeric_opts;
    }
    /// Clear all string options.
    void reset_string_options()
    {
        m_string_opts.clear();
    }
    /// Clear all integer options.
    void reset_integer_options()
    {
        m_integer_opts.clear();
    }
    /// Clear all numeric options.
    void reset_numeric_options()
    {
        m_numeric_opts.clear();
    }
    /// Thread safety level.
    /**
     * According to the official Ipopt documentation, it is not safe to use Ipopt in a multithreaded environment.
     *
     * @return thread_safety::none.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. seealso::
     *    https://projects.coin-or.org/Ipopt/wiki/FAQ
     *
     * \endverbatim
     */
    thread_safety get_thread_safety() const
    {
        return thread_safety::none;
    }

private:
    // Options maps.
    std::map<std::string, std::string> m_string_opts;
    std::map<std::string, Ipopt::Index> m_integer_opts;
    std::map<std::string, double> m_numeric_opts;
    // Solver return status.
    mutable Ipopt::ApplicationReturnStatus m_last_opt_res = Ipopt::Solve_Succeeded;
    // Verbosity/log.
    unsigned m_verbosity = 0;
    mutable log_type m_log;
};
} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::ipopt)

#else // PAGMO_WITH_IPOPT

#error The ipopt.hpp header was included, but pagmo was not compiled with Ipopt support

#endif // PAGMO_WITH_IPOPT

#endif
