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

#ifndef PAGMO_DETAIL_NLOPT_UTILS_HPP
#define PAGMO_DETAIL_NLOPT_UTILS_HPP

#include <algorithm>
#include <boost/bimap.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <nlopt.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

// Usual trick with global read-only data useful to the NLopt wrapper.
template <typename = void>
struct nlopt_data {
    // The idea here is to establish a bijection between string name (e.g., "cobyla")
    // and the enums used in the NLopt C API to refer to the algos (e.g., NLOPT_LN_COBYLA).
    // We use a bidirectional map so that we can map both string -> enum and enum -> string,
    // depending on what is needed.
    using names_map_t = boost::bimap<std::string, ::nlopt_algorithm>;
    static const names_map_t names;
    // A map to link a human-readable description to NLopt return codes.
    using result_map_t = std::unordered_map<::nlopt_result, std::string>;
    static const result_map_t results;
};

// Static init.
template <typename T>
const typename nlopt_data<T>::result_map_t nlopt_data<T>::results = {
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

// Initialise the mapping between algo names and enums for the supported algorithms.
inline typename nlopt_data<>::names_map_t nlopt_names_map()
{
    typename nlopt_data<>::names_map_t retval;
    using value_type = typename nlopt_data<>::names_map_t::value_type;
    retval.insert(value_type("cobyla", NLOPT_LN_COBYLA));
    retval.insert(value_type("bobyqa", NLOPT_LN_BOBYQA));
    retval.insert(value_type("praxis", NLOPT_LN_PRAXIS));
    retval.insert(value_type("neldermead", NLOPT_LN_NELDERMEAD));
    retval.insert(value_type("sbplx", NLOPT_LN_SBPLX));
    return retval;
}

// Static init using the helper function above.
template <typename T>
const typename nlopt_data<T>::names_map_t nlopt_data<T>::names = nlopt_names_map();

// Convert an NLopt result in a more descriptive string.
inline std::string nlopt_res2string(::nlopt_result err)
{
    return (nlopt_data<>::results.find(err) == nlopt_data<>::results.end() ? "??" : nlopt_data<>::results.at(err));
}

struct nlopt_obj {
    using data = nlopt_data<>;
    explicit nlopt_obj(::nlopt_algorithm algo, problem &prob, double stopval, double ftol_rel, double ftol_abs,
                       double xtol_rel, double xtol_abs, int maxeval, int maxtime)
        : m_prob(prob), m_sp(prob.gradient_sparsity()), m_value(nullptr, ::nlopt_destroy)
    {
        // Extract and set problem dimension.
        const auto n = boost::numeric_cast<unsigned>(prob.get_nx());
        m_value.reset(::nlopt_create(algo, n));
        // Try to init the nlopt_obj.
        if (!m_value) {
            pagmo_throw(std::invalid_argument, "the creation of the nlopt_opt object failed");
        }

        // NLopt does not handle MOO.
        if (prob.get_nobj() != 1u) {
            pagmo_throw(std::invalid_argument, "NLopt algorithms cannot handle multi-objective optimization");
        }

        // Constraints support will come later.
        if (prob.get_nc()) {
            // TODO error message
            pagmo_throw(std::invalid_argument, "");
        }

        ::nlopt_result res;

        // Box bounds.
        const auto bounds = prob.get_bounds();
        res = ::nlopt_set_lower_bounds(m_value.get(), bounds.first.data());
        if (res != NLOPT_SUCCESS) {
            pagmo_throw(std::invalid_argument, "could not set the lower bounds for the NLopt algorithm '"
                                                   + data::names.right.at(algo) + "', the error is: "
                                                   + nlopt_res2string(res));
        }
        res = ::nlopt_set_upper_bounds(m_value.get(), bounds.second.data());
        if (res != NLOPT_SUCCESS) {
            pagmo_throw(std::invalid_argument, "could not set the upper bounds for the NLopt algorithm '"
                                                   + data::names.right.at(algo) + "', the error is: "
                                                   + nlopt_res2string(res));
        }

        // This is just a vector_double that is re-used across objfun invocations.
        // It will hold the current decision vector.
        m_dv.resize(prob.get_nx());
        // Set the objfun + gradient.
        res = ::nlopt_set_min_objective(
            m_value.get(),
            [](unsigned dim, const double *x, double *grad, void *f_data) {
                // Get *this back from the function data.
                auto &nlo = *static_cast<nlopt_obj *>(f_data);

                // A few shortcuts.
                auto &p = nlo.m_prob;
                auto &dv = nlo.m_dv;
                auto &sp = nlo.m_sp;

                // A couple of sanity checks.
                assert(dim == p.get_nx());
                assert(dv.size() == dim);

                if (grad && !p.has_gradient()) {
                    // If grad is not null, it means we are in an algorithm
                    // that needs the gradient. If the problem does not support it,
                    // we error out.
                    pagmo_throw(std::invalid_argument,
                                "during an optimization with the NLopt algorithm '"
                                    + data::names.right.at(::nlopt_get_algorithm(nlo.m_value.get()))
                                    + "' a gradient was requested, but the optimisation problem '" + p.get_name()
                                    + "' does not provide it");
                }

                // Copy the decision vector in our temporary dv vector_double,
                // for use in the pagmo API.
                std::copy(x, x + dim, dv.begin());

                // Compute fitness and, if needed, gradient.
                const auto fitness = p.fitness(dv);
                if (grad) {
                    const auto gradient = p.gradient(dv);
                    auto g_it = gradient.begin();
                    // NOTE: problem::gradient() has already checked that
                    // the returned vector has size m_gs_dim, i.e., the stored
                    // size of the sparsity pattern. On the other hand,
                    // problem::gradient_sparsity() also checks that the returned
                    // vector has size m_gs_dim, so these two must have the same size.
                    assert(gradient.size() == sp.size());

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
                }

                // Return the objfun value.
                return fitness[0];
            },
            static_cast<void *>(this));
        if (res != NLOPT_SUCCESS) {
            pagmo_throw(std::invalid_argument, "could not set the objective function for the NLopt algorithm '"
                                                   + data::names.right.at(algo) + "', the error is: "
                                                   + nlopt_res2string(res));
        }

        // Handle the various stopping criteria.
        res = ::nlopt_set_stopval(m_value.get(), stopval);
        if (res != NLOPT_SUCCESS) {
            pagmo_throw(std::invalid_argument, "could not set the 'stopval' stopping criterion to "
                                                   + std::to_string(stopval) + " for the NLopt algorithm '"
                                                   + data::names.right.at(algo) + "', the error is: "
                                                   + nlopt_res2string(res));
        }
        res = ::nlopt_set_ftol_rel(m_value.get(), ftol_rel);
        if (res != NLOPT_SUCCESS) {
            pagmo_throw(std::invalid_argument, "could not set the 'ftol_rel' stopping criterion to "
                                                   + std::to_string(ftol_rel) + " for the NLopt algorithm '"
                                                   + data::names.right.at(algo) + "', the error is: "
                                                   + nlopt_res2string(res));
        }
        res = ::nlopt_set_ftol_abs(m_value.get(), ftol_abs);
        if (res != NLOPT_SUCCESS) {
            pagmo_throw(std::invalid_argument, "could not set the 'ftol_abs' stopping criterion to "
                                                   + std::to_string(ftol_abs) + " for the NLopt algorithm '"
                                                   + data::names.right.at(algo) + "', the error is: "
                                                   + nlopt_res2string(res));
        }
        res = ::nlopt_set_xtol_rel(m_value.get(), xtol_rel);
        if (res != NLOPT_SUCCESS) {
            pagmo_throw(std::invalid_argument, "could not set the 'xtol_rel' stopping criterion to "
                                                   + std::to_string(xtol_rel) + " for the NLopt algorithm '"
                                                   + data::names.right.at(algo) + "', the error is: "
                                                   + nlopt_res2string(res));
        }
        res = ::nlopt_set_xtol_abs1(m_value.get(), xtol_abs);
        if (res != NLOPT_SUCCESS) {
            pagmo_throw(std::invalid_argument, "could not set the 'xtol_abs' stopping criterion to "
                                                   + std::to_string(xtol_abs) + " for the NLopt algorithm '"
                                                   + data::names.right.at(algo) + "', the error is: "
                                                   + nlopt_res2string(res));
        }
        res = ::nlopt_set_maxeval(m_value.get(), maxeval);
        if (res != NLOPT_SUCCESS) {
            pagmo_throw(std::invalid_argument, "could not set the 'maxeval' stopping criterion to "
                                                   + std::to_string(maxeval) + " for the NLopt algorithm '"
                                                   + data::names.right.at(algo) + "', the error is: "
                                                   + nlopt_res2string(res));
        }
        res = ::nlopt_set_maxtime(m_value.get(), maxtime);
        if (res != NLOPT_SUCCESS) {
            pagmo_throw(std::invalid_argument, "could not set the 'maxtime' stopping criterion to "
                                                   + std::to_string(maxtime) + " for the NLopt algorithm '"
                                                   + data::names.right.at(algo) + "', the error is: "
                                                   + nlopt_res2string(res));
        }
    }
    nlopt_obj(const nlopt_obj &) = delete;
    nlopt_obj(nlopt_obj &&) = delete;
    nlopt_obj &operator=(const nlopt_obj &) = delete;
    nlopt_obj &operator=(nlopt_obj &&) = delete;

    // Data members.
    problem &m_prob;
    sparsity_pattern m_sp;
    std::unique_ptr<std::remove_pointer<::nlopt_opt>::type, void (*)(::nlopt_opt)> m_value;
    vector_double m_dv;
};
}
}

#endif
