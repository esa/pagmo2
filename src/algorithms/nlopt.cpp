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

#if defined(_MSC_VER)

// Disable the checked iterators feature in MSVC. There are some warnings
// triggered by Boost algos instantiations which we cannot do much about.
#define _SCL_SECURE_NO_WARNINGS

#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlopt.h>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/any.hpp>
#include <boost/bimap.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>

namespace pagmo
{

namespace detail
{

namespace
{

// The idea here is to establish a bijection between string name (e.g., "cobyla")
// and the enums used in the NLopt C API to refer to the algos (e.g., NLOPT_LN_COBYLA).
// We use a bidirectional map so that we can map both string -> enum and enum -> string,
// depending on what is needed.
using nlopt_names_map_t = boost::bimap<std::string, ::nlopt_algorithm>;

// Initialise the mapping between algo names and enums for the supported algorithms.
nlopt_names_map_t nlopt_names_map_init()
{
    nlopt_names_map_t retval;
    using value_type = nlopt_names_map_t::value_type;
    retval.insert(value_type("cobyla", NLOPT_LN_COBYLA));
    retval.insert(value_type("bobyqa", NLOPT_LN_BOBYQA));
    retval.insert(value_type("newuoa", NLOPT_LN_NEWUOA));
    retval.insert(value_type("newuoa_bound", NLOPT_LN_NEWUOA_BOUND));
    retval.insert(value_type("praxis", NLOPT_LN_PRAXIS));
    retval.insert(value_type("neldermead", NLOPT_LN_NELDERMEAD));
    retval.insert(value_type("sbplx", NLOPT_LN_SBPLX));
    retval.insert(value_type("mma", NLOPT_LD_MMA));
    retval.insert(value_type("ccsaq", NLOPT_LD_CCSAQ));
    retval.insert(value_type("slsqp", NLOPT_LD_SLSQP));
    retval.insert(value_type("lbfgs", NLOPT_LD_LBFGS));
    retval.insert(value_type("tnewton_precond_restart", NLOPT_LD_TNEWTON_PRECOND_RESTART));
    retval.insert(value_type("tnewton_precond", NLOPT_LD_TNEWTON_PRECOND));
    retval.insert(value_type("tnewton_restart", NLOPT_LD_TNEWTON_RESTART));
    retval.insert(value_type("tnewton", NLOPT_LD_TNEWTON));
    retval.insert(value_type("var2", NLOPT_LD_VAR2));
    retval.insert(value_type("var1", NLOPT_LD_VAR1));
    retval.insert(value_type("auglag", NLOPT_AUGLAG));
    retval.insert(value_type("auglag_eq", NLOPT_AUGLAG_EQ));
    return retval;
}

const nlopt_names_map_t nlopt_names = nlopt_names_map_init();

// A map to link a human-readable description to NLopt return codes.
// NOTE: in C++11 hashing of enums might not be available. Provide our own.
struct nlopt_res_hasher {
    std::size_t operator()(::nlopt_result res) const noexcept
    {
        return std::hash<int>{}(static_cast<int>(res));
    }
};

using nlopt_result_map_t = std::unordered_map<::nlopt_result, std::string, nlopt_res_hasher>;

const nlopt_result_map_t nlopt_results = {
    {NLOPT_SUCCESS, "NLOPT_SUCCESS (value = " + std::to_string(NLOPT_SUCCESS) + ", Generic success return value)"},
    {NLOPT_STOPVAL_REACHED, "NLOPT_STOPVAL_REACHED (value = " + std::to_string(NLOPT_STOPVAL_REACHED)
                                + ", Optimization stopped because stopval was reached)"},
    {NLOPT_FTOL_REACHED, "NLOPT_FTOL_REACHED (value = " + std::to_string(NLOPT_FTOL_REACHED)
                             + ", Optimization stopped because ftol_rel or ftol_abs was reached)"},
    {NLOPT_XTOL_REACHED, "NLOPT_XTOL_REACHED (value = " + std::to_string(NLOPT_XTOL_REACHED)
                             + ", Optimization stopped because xtol_rel or xtol_abs was reached)"},
    {NLOPT_MAXEVAL_REACHED, "NLOPT_MAXEVAL_REACHED (value = " + std::to_string(NLOPT_MAXEVAL_REACHED)
                                + ", Optimization stopped because maxeval was reached)"},
    {NLOPT_MAXTIME_REACHED, "NLOPT_MAXTIME_REACHED (value = " + std::to_string(NLOPT_MAXTIME_REACHED)
                                + ", Optimization stopped because maxtime was reached)"},
    {NLOPT_FAILURE, "NLOPT_FAILURE (value = " + std::to_string(NLOPT_FAILURE) + ", Generic failure code)"},
    {NLOPT_INVALID_ARGS, "NLOPT_INVALID_ARGS (value = " + std::to_string(NLOPT_INVALID_ARGS) + ", Invalid arguments)"},
    {NLOPT_OUT_OF_MEMORY,
     "NLOPT_OUT_OF_MEMORY (value = " + std::to_string(NLOPT_OUT_OF_MEMORY) + ", Ran out of memory)"},
    {NLOPT_ROUNDOFF_LIMITED, "NLOPT_ROUNDOFF_LIMITED (value = " + std::to_string(NLOPT_ROUNDOFF_LIMITED)
                                 + ", Halted because roundoff errors limited progress)"},
    {NLOPT_FORCED_STOP,
     "NLOPT_FORCED_STOP (value = " + std::to_string(NLOPT_FORCED_STOP) + ", Halted because of a forced termination)"}};

// Convert an NLopt result in a more descriptive string.
std::string nlopt_res2string(::nlopt_result err)
{
    return (nlopt_results.find(err) == nlopt_results.end() ? "??" : nlopt_results.at(err));
}

extern "C" {

// Wrappers to connect our objfun/constraints calculation machinery to NLopt's. Declared here,
// defined later in order to avoid circular deps.
// NOTE: these functions need to be passed to the NLopt C API, and as such they need to be
// declated within an 'extern "C"' block (otherwise, it might be UB to pass C++ function pointers
// to a C API).
// https://www.reddit.com/r/cpp/comments/4fqfy7/using_c11_capturing_lambdas_w_vanilla_c_api/d2b9bh0/
double nlopt_objfun_wrapper(unsigned, const double *, double *, void *);
void nlopt_ineq_c_wrapper(unsigned, double *, unsigned, const double *, double *, void *);
void nlopt_eq_c_wrapper(unsigned, double *, unsigned, const double *, double *, void *);
}

struct nlopt_obj {
    // Single entry of the log (objevals, objval, n of unsatisfied const, constr. violation, feasibility).
    // Same as defined in the nlopt algorithm.
    using log_line_type = nlopt::log_line_type;
    // The log.
    using log_type = std::vector<log_line_type>;
    // NOTE: this is a wrapper around std::copy() for use in MSVC in conjunction with raw pointers.
    // In debug mode, MSVC will complain about unchecked iterators unless instructed otherwise.
    template <typename Int, typename T>
    static void unchecked_copy(Int size, const T *begin, T *dest)
    {
        std::copy(begin, begin + size, dest);
    }
    explicit nlopt_obj(::nlopt_algorithm algo, problem &prob, double stopval, double ftol_rel, double ftol_abs,
                       double xtol_rel, double xtol_abs, int maxeval, int maxtime, unsigned verbosity)
        : m_algo(algo), m_prob(prob), m_value(nullptr, ::nlopt_destroy), m_verbosity(verbosity)
    {
        // Extract and set problem dimension.
        const auto n = boost::numeric_cast<unsigned>(prob.get_nx());
        // Try to init the nlopt_obj.
        m_value.reset(::nlopt_create(algo, n));
        if (!m_value) {
            pagmo_throw(std::runtime_error, "the creation of the nlopt_opt object failed"); // LCOV_EXCL_LINE
        }

        // NLopt does not handle MOO.
        if (prob.get_nobj() != 1u) {
            pagmo_throw(std::invalid_argument, "NLopt algorithms cannot handle multi-objective optimization");
        }

        // This is just a vector_double that is re-used across objfun invocations.
        // It will hold the current decision vector.
        m_dv.resize(prob.get_nx());

        // Handle the various stopping criteria.
        auto res = ::nlopt_set_stopval(m_value.get(), stopval);
        if (res != NLOPT_SUCCESS) {
            // LCOV_EXCL_START
            pagmo_throw(std::invalid_argument, "could not set the 'stopval' stopping criterion to "
                                                   + std::to_string(stopval) + " for the NLopt algorithm '"
                                                   + nlopt_names.right.at(algo)
                                                   + "', the error is: " + nlopt_res2string(res));
            // LCOV_EXCL_STOP
        }
        res = ::nlopt_set_ftol_rel(m_value.get(), ftol_rel);
        if (res != NLOPT_SUCCESS) {
            // LCOV_EXCL_START
            pagmo_throw(std::invalid_argument, "could not set the 'ftol_rel' stopping criterion to "
                                                   + std::to_string(ftol_rel) + " for the NLopt algorithm '"
                                                   + nlopt_names.right.at(algo)
                                                   + "', the error is: " + nlopt_res2string(res));
            // LCOV_EXCL_STOP
        }
        res = ::nlopt_set_ftol_abs(m_value.get(), ftol_abs);
        if (res != NLOPT_SUCCESS) {
            // LCOV_EXCL_START
            pagmo_throw(std::invalid_argument, "could not set the 'ftol_abs' stopping criterion to "
                                                   + std::to_string(ftol_abs) + " for the NLopt algorithm '"
                                                   + nlopt_names.right.at(algo)
                                                   + "', the error is: " + nlopt_res2string(res));
            // LCOV_EXCL_STOP
        }
        res = ::nlopt_set_xtol_rel(m_value.get(), xtol_rel);
        if (res != NLOPT_SUCCESS) {
            // LCOV_EXCL_START
            pagmo_throw(std::invalid_argument, "could not set the 'xtol_rel' stopping criterion to "
                                                   + std::to_string(xtol_rel) + " for the NLopt algorithm '"
                                                   + nlopt_names.right.at(algo)
                                                   + "', the error is: " + nlopt_res2string(res));
            // LCOV_EXCL_STOP
        }
        res = ::nlopt_set_xtol_abs1(m_value.get(), xtol_abs);
        if (res != NLOPT_SUCCESS) {
            // LCOV_EXCL_START
            pagmo_throw(std::invalid_argument, "could not set the 'xtol_abs' stopping criterion to "
                                                   + std::to_string(xtol_abs) + " for the NLopt algorithm '"
                                                   + nlopt_names.right.at(algo)
                                                   + "', the error is: " + nlopt_res2string(res));
            // LCOV_EXCL_STOP
        }
        res = ::nlopt_set_maxeval(m_value.get(), maxeval);
        if (res != NLOPT_SUCCESS) {
            // LCOV_EXCL_START
            pagmo_throw(std::invalid_argument, "could not set the 'maxeval' stopping criterion to "
                                                   + std::to_string(maxeval) + " for the NLopt algorithm '"
                                                   + nlopt_names.right.at(algo)
                                                   + "', the error is: " + nlopt_res2string(res));
            // LCOV_EXCL_STOP
        }
        res = ::nlopt_set_maxtime(m_value.get(), maxtime);
        if (res != NLOPT_SUCCESS) {
            // LCOV_EXCL_START
            pagmo_throw(std::invalid_argument, "could not set the 'maxtime' stopping criterion to "
                                                   + std::to_string(maxtime) + " for the NLopt algorithm '"
                                                   + nlopt_names.right.at(algo)
                                                   + "', the error is: " + nlopt_res2string(res));
            // LCOV_EXCL_STOP
        }
    }

    // Set box bounds.
    void set_bounds()
    {
        const auto bounds = m_prob.get_bounds();
        auto res = ::nlopt_set_lower_bounds(m_value.get(), bounds.first.data());
        if (res != NLOPT_SUCCESS) {
            // LCOV_EXCL_START
            pagmo_throw(std::invalid_argument, "could not set the lower bounds for the NLopt algorithm '"
                                                   + nlopt_names.right.at(m_algo)
                                                   + "', the error is: " + nlopt_res2string(res));
            // LCOV_EXCL_STOP
        }
        res = ::nlopt_set_upper_bounds(m_value.get(), bounds.second.data());
        if (res != NLOPT_SUCCESS) {
            // LCOV_EXCL_START
            pagmo_throw(std::invalid_argument, "could not set the upper bounds for the NLopt algorithm '"
                                                   + nlopt_names.right.at(m_algo)
                                                   + "', the error is: " + nlopt_res2string(res));
            // LCOV_EXCL_STOP
        }
    }

    // Set the objfun + gradient.
    void set_objfun()
    {
        // If needed, init the sparsity pattern.
        // NOTE: we do it here so that, in case this is a local optimiser,
        // we don't waste memory (set_objfun() etc. are not called when setting up a local
        // optimiser).
        if (m_prob.has_gradient_sparsity()) {
            m_sp = m_prob.gradient_sparsity();
        }
        auto res = ::nlopt_set_min_objective(m_value.get(), nlopt_objfun_wrapper, static_cast<void *>(this));
        if (res != NLOPT_SUCCESS) {
            // LCOV_EXCL_START
            pagmo_throw(std::invalid_argument, "could not set the objective function for the NLopt algorithm '"
                                                   + nlopt_names.right.at(m_algo)
                                                   + "', the error is: " + nlopt_res2string(res));
            // LCOV_EXCL_STOP
        }
    }

    // Inequality constraints.
    void set_ineq_constraints()
    {
        if (m_prob.get_nic()) {
            const auto c_tol = m_prob.get_c_tol();
            auto res = ::nlopt_add_inequality_mconstraint(
                m_value.get(), boost::numeric_cast<unsigned>(m_prob.get_nic()), nlopt_ineq_c_wrapper,
                static_cast<void *>(this), c_tol.data() + m_prob.get_nec());
            if (res != NLOPT_SUCCESS) {
                pagmo_throw(std::invalid_argument,
                            "could not set the inequality constraints for the NLopt algorithm '"
                                + nlopt_names.right.at(m_algo) + "', the error is: " + nlopt_res2string(res)
                                + "\nThis usually means that the algorithm does not support inequality constraints");
            }
        }
    }

    // Equality constraints.
    void set_eq_constraints()
    {
        if (m_prob.get_nec()) {
            const auto c_tol = m_prob.get_c_tol();
            auto res = ::nlopt_add_equality_mconstraint(m_value.get(), boost::numeric_cast<unsigned>(m_prob.get_nec()),
                                                        nlopt_eq_c_wrapper, static_cast<void *>(this), c_tol.data());
            if (res != NLOPT_SUCCESS) {
                pagmo_throw(std::invalid_argument,
                            "could not set the equality constraints for the NLopt algorithm '"
                                + nlopt_names.right.at(m_algo) + "', the error is: " + nlopt_res2string(res)
                                + "\nThis usually means that the algorithm does not support equality constraints");
            }
        }
    }

    // Delete all other ctors/assignment ops.
    nlopt_obj(const nlopt_obj &) = delete;
    nlopt_obj(nlopt_obj &&) = delete;
    nlopt_obj &operator=(const nlopt_obj &) = delete;
    nlopt_obj &operator=(nlopt_obj &&) = delete;

    // Data members.
    ::nlopt_algorithm m_algo;
    problem &m_prob;
    sparsity_pattern m_sp;
    std::unique_ptr<std::remove_pointer<::nlopt_opt>::type, void (*)(::nlopt_opt)> m_value;
    vector_double m_dv;
    unsigned m_verbosity;
    unsigned long m_objfun_counter = 0;
    log_type m_log;
    // This exception pointer will be null, unless
    // an error is raised during the computation of the objfun
    // or constraints. If not null, it will be re-thrown
    // in the evolve() method.
    std::exception_ptr m_eptr;
};

double nlopt_objfun_wrapper(unsigned dim, const double *x, double *grad, void *f_data)
{
    // Get *this back from the function data.
    auto &nlo = *static_cast<nlopt_obj *>(f_data);

    // NOTE: the idea here is that we wrap everything in a try/catch block,
    // and, if any exception is thrown, we record it into the nlo object
    // and re-throw it later. We do this because we are using the NLopt C API,
    // and if we let exceptions out of here we run in undefined behaviour.
    // We do the same for the constraints functions.
    try {
        // A few shortcuts.
        auto &p = nlo.m_prob;
        auto &dv = nlo.m_dv;
        const auto verb = nlo.m_verbosity;
        auto &f_count = nlo.m_objfun_counter;
        auto &log = nlo.m_log;

        // A couple of sanity checks.
        assert(dim == p.get_nx());
        assert(dv.size() == dim);

        if (grad && !p.has_gradient()) {
            // If grad is not null, it means we are in an algorithm
            // that needs the gradient. If the problem does not support it,
            // we error out.
            pagmo_throw(std::invalid_argument,
                        "during an optimization with the NLopt algorithm '"
                            + nlopt_names.right.at(::nlopt_get_algorithm(nlo.m_value.get()))
                            + "' an objective function gradient was requested, but the optimisation problem '"
                            + p.get_name() + "' does not provide it");
        }

        // Copy the decision vector in our temporary dv vector_double,
        // for use in the pagmo API.
        std::copy(x, x + dim, dv.begin());

        // Compute fitness.
        const auto fitness = p.fitness(dv);

        // Compute gradient, if needed.
        if (grad) {
            const auto gradient = p.gradient(dv);

            if (p.has_gradient_sparsity()) {
                // Sparse gradient case.
                const auto &sp = nlo.m_sp;
                // NOTE: problem::gradient() has already checked that
                // the returned vector has size m_gs_dim, i.e., the stored
                // size of the sparsity pattern. On the other hand,
                // problem::gradient_sparsity() also checks that the returned
                // vector has size m_gs_dim, so these two must have the same size.
                assert(gradient.size() == sp.size());
                auto g_it = gradient.begin();

                // First we fill the dense output gradient with zeroes.
                std::fill(grad, grad + dim, 0.);
                // Then we iterate over the sparsity pattern, and fill in the
                // nonzero bits in grad.
                for (auto it = sp.begin(); it != sp.end() && it->first == 0u; ++it, ++g_it) {
                    // NOTE: we just need the gradient of the objfun,
                    // i.e., those (i,j) pairs in which i == 0. We know that the gradient
                    // of the objfun, if present, starts at the beginning of sp, as sp is
                    // sorted in lexicographic fashion.
                    grad[it->second] = *g_it;
                }
            } else {
                // Dense gradient case.
                nlopt_obj::unchecked_copy(p.get_nx(), gradient.data(), grad);
            }
        }

        // Update the log if requested.
        if (verb && !(f_count % verb)) {
            // Constraints bits.
            const auto ctol = p.get_c_tol();
            const auto c1eq
                = detail::test_eq_constraints(fitness.data() + 1, fitness.data() + 1 + p.get_nec(), ctol.data());
            const auto c1ineq = detail::test_ineq_constraints(
                fitness.data() + 1 + p.get_nec(), fitness.data() + fitness.size(), ctol.data() + p.get_nec());
            // This will be the total number of violated constraints.
            const auto nv = p.get_nc() - c1eq.first - c1ineq.first;
            // This will be the norm of the violation.
            const auto l = c1eq.second + c1ineq.second;
            // Test feasibility.
            const auto feas = p.feasibility_f(fitness);

            if (!(f_count / verb % 50u)) {
                // Every 50 lines print the column names.
                print("\n", std::setw(10), "objevals:", std::setw(15), "objval:", std::setw(15),
                      "violated:", std::setw(15), "viol. norm:", '\n');
            }
            // Print to screen the log line.
            print(std::setw(10), f_count + 1u, std::setw(15), fitness[0], std::setw(15), nv, std::setw(15), l,
                  feas ? "" : " i", '\n');
            // Record the log.
            log.emplace_back(f_count + 1u, fitness[0], nv, l, feas);
        }

        // Update the counter.
        ++f_count;

        // Return the objfun value.
        return fitness[0];
    } catch (...) {
        // Store exception, force the stop of the optimisation,
        // and return a useless value.
        nlo.m_eptr = std::current_exception();
        ::nlopt_force_stop(nlo.m_value.get());
        return HUGE_VAL;
    }
}

void nlopt_ineq_c_wrapper(unsigned m, double *result, unsigned dim, const double *x, double *grad, void *f_data)
{
    // Get *this back from the function data.
    auto &nlo = *static_cast<nlopt_obj *>(f_data);

    try {
        // A few shortcuts.
        auto &p = nlo.m_prob;
        auto &dv = nlo.m_dv;

        // A couple of sanity checks.
        assert(dim == p.get_nx());
        assert(dv.size() == dim);
        assert(m == p.get_nic());
        (void)m;

        if (grad && !p.has_gradient()) {
            // If grad is not null, it means we are in an algorithm
            // that needs the gradient. If the problem does not support it,
            // we error out.
            pagmo_throw(std::invalid_argument, "during an optimization with the NLopt algorithm '"
                                                   + nlopt_names.right.at(::nlopt_get_algorithm(nlo.m_value.get()))
                                                   + "' an inequality constraints gradient was requested, but the "
                                                     "optimisation problem '"
                                                   + p.get_name() + "' does not provide it");
        }

        // Copy the decision vector in our temporary dv vector_double,
        // for use in the pagmo API.
        std::copy(x, x + dim, dv.begin());

        // Compute fitness and write IC to the output.
        // NOTE: fitness is nobj + nec + nic.
        const auto fitness = p.fitness(dv);
        nlopt_obj::unchecked_copy(p.get_nic(), fitness.data() + 1 + p.get_nec(), result);

        if (grad) {
            // Handle gradient, if requested.
            const auto gradient = p.gradient(dv);

            if (p.has_gradient_sparsity()) {
                // Sparse gradient.
                const auto &sp = nlo.m_sp;
                // NOTE: problem::gradient() has already checked that
                // the returned vector has size m_gs_dim, i.e., the stored
                // size of the sparsity pattern. On the other hand,
                // problem::gradient_sparsity() also checks that the returned
                // vector has size m_gs_dim, so these two must have the same size.
                assert(gradient.size() == sp.size());

                // Let's first fill it with zeroes.
                std::fill(grad, grad + p.get_nx() * p.get_nic(), 0.);

                // Now we need to go into the sparsity pattern and find where
                // the sparsity data for the constraints start.
                auto it_sp = std::lower_bound(sp.begin(), sp.end(), sparsity_pattern::value_type(p.get_nec() + 1u, 0u));

                // Need to do a bit of horrid overflow checking :/.
                using diff_type = std::iterator_traits<decltype(it_sp)>::difference_type;
                using udiff_type = std::make_unsigned<diff_type>::type;
                if (sp.size() > static_cast<udiff_type>(std::numeric_limits<diff_type>::max())) {
                    pagmo_throw(std::overflow_error, "Overflow error, the sparsity pattern size is too large.");
                }
                // This is the index at which the ineq constraints start.
                const auto idx = std::distance(sp.begin(), it_sp);
                // Grab the start of the gradient data for the ineq constraints.
                auto g_it = gradient.data() + idx;

                // Then we iterate over the sparsity pattern, and fill in the
                // nonzero bits in grad. Run until sp.end() as the IC are at the
                // end of the sparsity/gradient vector.
                for (; it_sp != sp.end(); ++it_sp, ++g_it) {
                    grad[(it_sp->first - 1u - p.get_nec()) * p.get_nx() + it_sp->second] = *g_it;
                }
            } else {
                // Dense gradient.
                nlopt_obj::unchecked_copy(p.get_nic() * p.get_nx(), gradient.data() + p.get_nx() * (1u + p.get_nec()),
                                          grad);
            }
        }
    } catch (...) {
        // Store exception, stop optimisation.
        nlo.m_eptr = std::current_exception();
        ::nlopt_force_stop(nlo.m_value.get());
    }
}

void nlopt_eq_c_wrapper(unsigned m, double *result, unsigned dim, const double *x, double *grad, void *f_data)
{
    // Get *this back from the function data.
    auto &nlo = *static_cast<nlopt_obj *>(f_data);

    try {
        // A few shortcuts.
        auto &p = nlo.m_prob;
        auto &dv = nlo.m_dv;

        // A couple of sanity checks.
        assert(dim == p.get_nx());
        assert(dv.size() == dim);
        assert(m == p.get_nec());

        if (grad && !p.has_gradient()) {
            // If grad is not null, it means we are in an algorithm
            // that needs the gradient. If the problem does not support it,
            // we error out.
            pagmo_throw(std::invalid_argument,
                        "during an optimization with the NLopt algorithm '"
                            + nlopt_names.right.at(::nlopt_get_algorithm(nlo.m_value.get()))
                            + "' an equality constraints gradient was requested, but the optimisation problem '"
                            + p.get_name() + "' does not provide it");
        }

        // Copy the decision vector in our temporary dv vector_double,
        // for use in the pagmo API.
        std::copy(x, x + dim, dv.begin());

        // Compute fitness and write EC to the output.
        // NOTE: fitness is nobj + nec + nic.
        const auto fitness = p.fitness(dv);
        nlopt_obj::unchecked_copy(p.get_nec(), fitness.data() + 1, result);

        if (grad) {
            // Handle gradient, if requested.
            const auto gradient = p.gradient(dv);

            if (p.has_gradient_sparsity()) {
                // Sparse gradient case.
                const auto &sp = nlo.m_sp;
                // NOTE: problem::gradient() has already checked that
                // the returned vector has size m_gs_dim, i.e., the stored
                // size of the sparsity pattern. On the other hand,
                // problem::gradient_sparsity() also checks that the returned
                // vector has size m_gs_dim, so these two must have the same size.
                assert(gradient.size() == sp.size());

                // Let's first fill it with zeroes.
                std::fill(grad, grad + p.get_nx() * p.get_nec(), 0.);

                // Now we need to go into the sparsity pattern and find where
                // the sparsity data for the constraints start.
                // NOTE: it_sp could be end() or point to ineq constraints. This should
                // be fine: it_sp is a valid iterator in sp, sp has the same
                // size as gradient and we do the proper checks below before accessing
                // the values pointed to by it_sp/g_it.
                auto it_sp = std::lower_bound(sp.begin(), sp.end(), sparsity_pattern::value_type(1u, 0u));

                // Need to do a bit of horrid overflow checking :/.
                using diff_type = std::iterator_traits<decltype(it_sp)>::difference_type;
                using udiff_type = std::make_unsigned<diff_type>::type;
                if (sp.size() > static_cast<udiff_type>(std::numeric_limits<diff_type>::max())) {
                    pagmo_throw(std::overflow_error, "Overflow error, the sparsity pattern size is too large.");
                }
                // This is the index at which the eq constraints start.
                const auto idx = std::distance(sp.begin(), it_sp);
                // Grab the start of the gradient data for the eq constraints.
                auto g_it = gradient.data() + idx;

                // Then we iterate over the sparsity pattern, and fill in the
                // nonzero bits in grad. We terminate either at the end of sp, or when
                // we encounter the first inequality constraint.
                for (; it_sp != sp.end() && it_sp->first < p.get_nec() + 1u; ++it_sp, ++g_it) {
                    grad[(it_sp->first - 1u) * p.get_nx() + it_sp->second] = *g_it;
                }
            } else {
                // Dense gradient.
                nlopt_obj::unchecked_copy(p.get_nx() * p.get_nec(), gradient.data() + p.get_nx(), grad);
            }
        }
    } catch (...) {
        // Store exception, stop optimisation.
        nlo.m_eptr = std::current_exception();
        ::nlopt_force_stop(nlo.m_value.get());
    }
}

} // namespace

} // namespace detail

/// Default constructor.
/**
 * The default constructor initialises the pagmo::nlopt algorithm with the ``"cobyla"`` solver.
 * The individual selection/replacement strategies are those specified by
 * not_population_based::not_population_based().
 *
 * @throws unspecified any exception thrown by pagmo::nlopt(const std::string &).
 */
nlopt::nlopt() : nlopt("cobyla") {}

/// Constructor from solver name.
/**
 * This constructor will initialise a pagmo::nlopt object which will use the NLopt algorithm specified by
 * the input string \p algo. The individual selection/replacement strategies are those specified by
 * not_population_based::not_population_based(). \p algo is translated to an NLopt algorithm type according to the
 * following table:
 * \verbatim embed:rst:leading-asterisk
 *  ================================  ====================================
 *  ``algo`` string                   NLopt algorithm
 *  ================================  ====================================
 *  ``"cobyla"``                      ``NLOPT_LN_COBYLA``
 *  ``"bobyqa"``                      ``NLOPT_LN_BOBYQA``
 *  ``"newuoa"``                      ``NLOPT_LN_NEWUOA``
 *  ``"newuoa_bound"``                ``NLOPT_LN_NEWUOA_BOUND``
 *  ``"praxis"``                      ``NLOPT_LN_PRAXIS``
 *  ``"neldermead"``                  ``NLOPT_LN_NELDERMEAD``
 *  ``"sbplx"``                       ``NLOPT_LN_SBPLX``
 *  ``"mma"``                         ``NLOPT_LD_MMA``
 *  ``"ccsaq"``                       ``NLOPT_LD_CCSAQ``
 *  ``"slsqp"``                       ``NLOPT_LD_SLSQP``
 *  ``"lbfgs"``                       ``NLOPT_LD_LBFGS``
 *  ``"tnewton_precond_restart"``     ``NLOPT_LD_TNEWTON_PRECOND_RESTART``
 *  ``"tnewton_precond"``             ``NLOPT_LD_TNEWTON_PRECOND``
 *  ``"tnewton_restart"``             ``NLOPT_LD_TNEWTON_RESTART``
 *  ``"tnewton"``                     ``NLOPT_LD_TNEWTON``
 *  ``"var2"``                        ``NLOPT_LD_VAR2``
 *  ``"var1"``                        ``NLOPT_LD_VAR1``
 *  ``"auglag"``                      ``NLOPT_AUGLAG``
 *  ``"auglag_eq"``                   ``NLOPT_AUGLAG_EQ``
 *  ================================  ====================================
 * \endverbatim
 * The parameters of the selected algorithm can be specified via the methods of this class.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. seealso::
 *
 *    The `NLopt website <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`__ contains a detailed
 *    description of each supported solver.
 *
 * \endverbatim
 *
 * @param algo the name of the NLopt algorithm that will be used by this pagmo::nlopt object.
 *
 * @throws std::runtime_error if the NLopt version is not at least 2.
 * @throws std::invalid_argument if \p algo is not one of the allowed algorithm names.
 * @throws unspecified any exception thrown by not_population_based::not_population_based().
 */
nlopt::nlopt(const std::string &algo) : m_algo(algo)
{
    // Check version.
    int major, minor, bugfix;
    ::nlopt_version(&major, &minor, &bugfix);
    if (major < 2) {
        pagmo_throw(std::runtime_error, "Only NLopt version >= 2 is supported"); // LCOV_EXCL_LINE
    }

    // Check the algorithm.
    if (detail::nlopt_names.left.find(m_algo) == detail::nlopt_names.left.end()) {
        // The selected algorithm is unknown or not among the supported ones.
        std::ostringstream oss;
        std::transform(detail::nlopt_names.left.begin(), detail::nlopt_names.left.end(),
                       std::ostream_iterator<std::string>(oss, "\n"),
                       [](const uncvref_t<decltype(*detail::nlopt_names.left.begin())> &v) { return v.first; });
        pagmo_throw(std::invalid_argument,
                    "unknown/unsupported NLopt algorithm '" + algo + "'. The supported algorithms are:\n" + oss.str());
    }
}

/// Copy constructor.
/**
 * The copy constructor will deep-copy the state of \p other.
 *
 * @param other the construction argument.
 *
 * @throws unspecified any exception thrown by copying the internal state of \p other.
 */
nlopt::nlopt(const nlopt &other)
    : not_population_based(other), m_algo(other.m_algo), m_last_opt_result(other.m_last_opt_result),
      m_sc_stopval(other.m_sc_stopval), m_sc_ftol_rel(other.m_sc_ftol_rel), m_sc_ftol_abs(other.m_sc_ftol_abs),
      m_sc_xtol_rel(other.m_sc_xtol_rel), m_sc_xtol_abs(other.m_sc_xtol_abs), m_sc_maxeval(other.m_sc_maxeval),
      m_sc_maxtime(other.m_sc_maxtime), m_verbosity(other.m_verbosity), m_log(other.m_log),
      m_loc_opt(other.m_loc_opt ? detail::make_unique<nlopt>(*other.m_loc_opt) : nullptr)
{
}

/// Evolve population.
/**
 * This method will select an individual from \p pop, optimise it with the NLopt algorithm specified upon
 * construction, replace an individual in \p pop with the optimised individual, and finally return \p pop.
 * The individual selection and replacement criteria can be set via set_selection(const std::string &),
 * set_selection(population::size_type), set_replacement(const std::string &) and
 * set_replacement(population::size_type). The NLopt solver will run until one of the stopping criteria
 * is satisfied, and the return status of the NLopt solver will be recorded (it can be fetched with
 * get_last_opt_result()).
 *
 * @param pop the population to be optimised.
 *
 * @return the optimised population.
 *
 * @throws std::invalid_argument in the following cases:
 * - the population's problem is multi-objective,
 * - the setup of the NLopt algorithm fails (e.g., if the problem is constrained but the selected
 *   NLopt solver does not support constrained optimisation),
 * - the selected NLopt solver needs gradients but they are not provided by the population's
 *   problem,
 * - the components of the individual selected for optimisation contain NaNs or they are outside
 *   the problem's bounds.
 * @throws unspecified any exception thrown by the public interface of pagmo::problem or
 * pagmo::not_population_based.
 */
population nlopt::evolve(population pop) const
{
    if (!pop.size()) {
        // In case of an empty pop, just return it.
        return pop;
    }

    auto &prob = pop.get_problem();

    // Create the nlopt obj.
    // NOTE: this will check also the problem's properties.
    detail::nlopt_obj no(detail::nlopt_names.left.at(m_algo), prob, m_sc_stopval, m_sc_ftol_rel, m_sc_ftol_abs,
                         m_sc_xtol_rel, m_sc_xtol_abs, m_sc_maxeval, m_sc_maxtime, m_verbosity);
    no.set_bounds();
    no.set_objfun();
    no.set_eq_constraints();
    no.set_ineq_constraints();

    // Set the local optimiser, if appropriate.
    if (m_loc_opt) {
        detail::nlopt_obj no_loc(detail::nlopt_names.left.at(m_loc_opt->m_algo), prob, m_loc_opt->m_sc_stopval,
                                 m_loc_opt->m_sc_ftol_rel, m_loc_opt->m_sc_ftol_abs, m_loc_opt->m_sc_xtol_rel,
                                 m_loc_opt->m_sc_xtol_abs, m_loc_opt->m_sc_maxeval, m_loc_opt->m_sc_maxtime, 0);
        ::nlopt_set_local_optimizer(no.m_value.get(), no_loc.m_value.get());
    }

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

    // Run the optimisation and store the status returned by NLopt.
    double objval;
    m_last_opt_result = ::nlopt_optimize(no.m_value.get(), initial_guess.data(), &objval);
    if (m_verbosity) {
        // Print to screen the result of the optimisation, if we are being verbose.
        std::cout << "\nOptimisation return status: " << detail::nlopt_res2string(m_last_opt_result) << '\n';
    }
    // Replace the log.
    m_log = std::move(no.m_log);

    // Handle any exception that might've been thrown.
    if (no.m_eptr) {
        std::rethrow_exception(no.m_eptr);
    }

    // Compute the new fitness vector.
    const auto new_f = prob.fitness(initial_guess);

    // Store the new individual into the population, but only if better.
    if (compare_fc(new_f, old_f, prob.get_nec(), prob.get_c_tol())) {
        replace_individual(pop, initial_guess, new_f);
    }

    // Return the evolved pop.
    return pop;
}

/// Algorithm's name.
/**
 * @return a human-readable name for the algorithm.
 */
std::string nlopt::get_name() const
{
    return "NLopt - " + m_algo + ":";
}

/// Get extra information about the algorithm.
/**
 * @return a human-readable string containing useful information about the algorithm's properties
 * (e.g., the stopping criteria, the selection/replacement policies, etc.).
 */
std::string nlopt::get_extra_info() const
{
    int major, minor, bugfix;
    ::nlopt_version(&major, &minor, &bugfix);
    auto retval = "\tNLopt version: " + std::to_string(major) + "." + std::to_string(minor) + "."
                  + std::to_string(bugfix) + "\n\tSolver: '" + m_algo
                  + "'\n\tLast optimisation return code: " + detail::nlopt_res2string(m_last_opt_result)
                  + "\n\tVerbosity: " + std::to_string(m_verbosity) + "\n\tIndividual selection "
                  + (boost::any_cast<population::size_type>(&m_select)
                         ? "idx: " + std::to_string(boost::any_cast<population::size_type>(m_select))
                         : "policy: " + boost::any_cast<std::string>(m_select))
                  + "\n\tIndividual replacement "
                  + (boost::any_cast<population::size_type>(&m_replace)
                         ? "idx: " + std::to_string(boost::any_cast<population::size_type>(m_replace))
                         : "policy: " + boost::any_cast<std::string>(m_replace))
                  + "\n\tStopping criteria:\n\t\tstopval:  "
                  + (m_sc_stopval == -HUGE_VAL ? "disabled" : detail::to_string(m_sc_stopval))
                  + "\n\t\tftol_rel: " + (m_sc_ftol_rel <= 0. ? "disabled" : detail::to_string(m_sc_ftol_rel))
                  + "\n\t\tftol_abs: " + (m_sc_ftol_abs <= 0. ? "disabled" : detail::to_string(m_sc_ftol_abs))
                  + "\n\t\txtol_rel: " + (m_sc_xtol_rel <= 0. ? "disabled" : detail::to_string(m_sc_xtol_rel))
                  + "\n\t\txtol_abs: " + (m_sc_xtol_abs <= 0. ? "disabled" : detail::to_string(m_sc_xtol_abs))
                  + "\n\t\tmaxeval:  " + (m_sc_maxeval <= 0. ? "disabled" : detail::to_string(m_sc_maxeval))
                  + "\n\t\tmaxtime:  " + (m_sc_maxtime <= 0. ? "disabled" : detail::to_string(m_sc_maxtime)) + "\n";
    if (m_loc_opt) {
        // Add a tab to the output of the extra_info() of the local opt,
        // and append the result.
        retval += "\tLocal optimizer:\n";
        const auto loc_info = m_loc_opt->get_extra_info();
        std::vector<std::string> split_v;
        boost::algorithm::split(split_v, loc_info, boost::algorithm::is_any_of("\n"),
                                boost::algorithm::token_compress_on);
        for (const auto &s : split_v) {
            retval += "\t" + s + "\n";
        }
    }
    return retval;
}

/// Set the ``stopval`` stopping criterion.
/**
 * @param stopval the desired value for the ``stopval`` stopping criterion (see get_stopval()).
 *
 * @throws std::invalid_argument if \p stopval is NaN.
 */
void nlopt::set_stopval(double stopval)
{
    if (std::isnan(stopval)) {
        pagmo_throw(std::invalid_argument, "The 'stopval' stopping criterion cannot be NaN");
    }
    m_sc_stopval = stopval;
}

/// Set the ``ftol_rel`` stopping criterion.
/**
 * @param ftol_rel the desired value for the ``ftol_rel`` stopping criterion (see get_ftol_rel()).
 *
 * @throws std::invalid_argument if \p ftol_rel is NaN.
 */
void nlopt::set_ftol_rel(double ftol_rel)
{
    if (std::isnan(ftol_rel)) {
        pagmo_throw(std::invalid_argument, "The 'ftol_rel' stopping criterion cannot be NaN");
    }
    m_sc_ftol_rel = ftol_rel;
}

/// Set the ``ftol_abs`` stopping criterion.
/**
 * @param ftol_abs the desired value for the ``ftol_abs`` stopping criterion (see get_ftol_abs()).
 *
 * @throws std::invalid_argument if \p ftol_abs is NaN.
 */
void nlopt::set_ftol_abs(double ftol_abs)
{
    if (std::isnan(ftol_abs)) {
        pagmo_throw(std::invalid_argument, "The 'ftol_abs' stopping criterion cannot be NaN");
    }
    m_sc_ftol_abs = ftol_abs;
}

/// Set the ``xtol_rel`` stopping criterion.
/**
 * @param xtol_rel the desired value for the ``xtol_rel`` stopping criterion (see get_xtol_rel()).
 *
 * @throws std::invalid_argument if \p xtol_rel is NaN.
 */
void nlopt::set_xtol_rel(double xtol_rel)
{
    if (std::isnan(xtol_rel)) {
        pagmo_throw(std::invalid_argument, "The 'xtol_rel' stopping criterion cannot be NaN");
    }
    m_sc_xtol_rel = xtol_rel;
}

/// Set the ``xtol_abs`` stopping criterion.
/**
 * @param xtol_abs the desired value for the ``xtol_abs`` stopping criterion (see get_xtol_abs()).
 *
 * @throws std::invalid_argument if \p xtol_abs is NaN.
 */
void nlopt::set_xtol_abs(double xtol_abs)
{
    if (std::isnan(xtol_abs)) {
        pagmo_throw(std::invalid_argument, "The 'xtol_abs' stopping criterion cannot be NaN");
    }
    m_sc_xtol_abs = xtol_abs;
}

/// Set the local optimizer.
/**
 * Some NLopt algorithms rely on other NLopt algorithms as local/subsidiary optimizers.
 * This method allows to set such local optimizer. By default, no local optimizer is specified.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    At the present time, only the ``"auglag"`` and ``"auglag_eq"`` solvers make use
 *    of a local optimizer. Setting a local optimizer on any other solver will have no effect.
 *
 * .. note::
 *
 *    The objective function, bounds, and nonlinear-constraint parameters of the local
 *    optimizer are ignored (as they are provided by the parent optimizer). Conversely, the stopping
 *    criteria should be specified in the local optimizer. The verbosity of
 *    the local optimizer is also forcibly set to zero during the optimisation.
 *
 * \endverbatim
 *
 * @param n the local optimizer that will be used by this pagmo::nlopt algorithm.
 */
void nlopt::set_local_optimizer(nlopt n)
{
    m_loc_opt = detail::make_unique<nlopt>(std::move(n));
}

/// Unset the local optimizer.
/**
 * After a call to this method, get_local_optimizer() and get_local_optimizer() const will return \p nullptr.
 */
void nlopt::unset_local_optimizer()
{
    m_loc_opt.reset(nullptr);
}

/// Save to archive.
/**
 * @param ar the target archive.
 *
 * @throws unspecified any exception thrown by the serialization of primitive types or pagmo::not_population_based.
 */
template <typename Archive>
void nlopt::save(Archive &ar, unsigned) const
{
    detail::to_archive(ar, boost::serialization::base_object<not_population_based>(*this), m_algo, m_last_opt_result,
                       m_sc_stopval, m_sc_ftol_rel, m_sc_ftol_abs, m_sc_xtol_rel, m_sc_xtol_abs, m_sc_maxeval,
                       m_sc_maxtime, m_verbosity, m_log);
    if (m_loc_opt) {
        detail::to_archive(ar, true, *m_loc_opt);
    } else {
        ar << false;
    }
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
void nlopt::load(Archive &ar, unsigned)
{
    try {
        detail::from_archive(ar, boost::serialization::base_object<not_population_based>(*this), m_algo,
                             m_last_opt_result, m_sc_stopval, m_sc_ftol_rel, m_sc_ftol_abs, m_sc_xtol_rel,
                             m_sc_xtol_abs, m_sc_maxeval, m_sc_maxtime, m_verbosity, m_log);
        bool with_local;
        ar >> with_local;
        if (with_local) {
            m_loc_opt = detail::make_unique<nlopt>();
            ar >> *m_loc_opt;
        }
    } catch (...) {
        *this = nlopt{};
        throw;
    }
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::nlopt)
