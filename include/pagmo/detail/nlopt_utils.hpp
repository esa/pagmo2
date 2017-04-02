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
#include <boost/numeric/conversion/cast.hpp>
#include <cassert>
#include <memory>
#include <nlopt.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

struct nlopt_obj {
    explicit nlopt_obj(::nlopt_algorithm algo, problem &prob)
        : m_prob(prob), m_sp(prob.gradient_sparsity()), m_value(nullptr, ::nlopt_destroy)
    {
        // Extract and set problem dimension.
        const auto n = boost::numeric_cast<unsigned>(prob.get_nx());
        m_value.reset(::nlopt_create(algo, n));
        // Try to init the nlopt_obj.
        if (!m_value) {
            pagmo_throw(std::invalid_argument, "the creation of an nlopt_opt object failed");
        }
        // NLopt does not handle MOO.
        if (prob.get_nobj() != 1u) {
            // TODO error message
            pagmo_throw(std::invalid_argument, "" + get_name());
        }
        // Constraints support will come later.
        if (prob.get_nc()) {
            // TODO error message
            pagmo_throw(std::invalid_argument, "" + get_name());
        }
        // This is just a vector_double that is re-used across objfun invocations.
        // It will hold the current decision vector.
        m_dv.resize(prob.get_nx());
        // Set the objfun + gradient.
        auto res = ::nlopt_set_min_objective(m_value.get(),
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
                                                     // TODO error message
                                                     pagmo_throw(std::invalid_argument, "" + nlo.get_name());
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
                                                     for (const auto &t : sp) {
                                                         if (t.first == 0u) {
                                                             // NOTE: we just need the gradient of the objfun,
                                                             // i.e., those (i,j) pairs in which i == 0.
                                                             grad[t.second] = *g_it;
                                                             ++g_it;
                                                         } else {
                                                             break;
                                                         }
                                                     }
                                                 }

                                                 // Return the objfun value.
                                                 return fitness[0];
                                             },
                                             static_cast<void *>(this));
        if (res != NLOPT_SUCCESS) {
            // TODO
            throw;
        }

        // Box bounds.
        const auto bounds = prob.get_bounds();
        res = ::nlopt_set_lower_bounds(m_value.get(), bounds.first.data());
        if (res != NLOPT_SUCCESS) {
            // TODO
            throw;
        }
        res = ::nlopt_set_upper_bounds(m_value.get(), bounds.second.data());
        if (res != NLOPT_SUCCESS) {
            // TODO
            throw;
        }

        // TODO hard-coded.
        res = ::nlopt_set_ftol_abs(m_value.get(), 1E-12);
        // res = ::nlopt_set_maxeval(m_value.get(), 10000);
        if (res != NLOPT_SUCCESS) {
            // TODO
            throw;
        }
    }
    nlopt_obj(const nlopt_obj &other)
        : m_prob(other.m_prob), m_sp(other.m_sp), m_value(::nlopt_copy(other.m_value.get()), ::nlopt_destroy),
          m_dv(other.m_dv)
    {
        if (!m_value) {
            pagmo_throw(std::invalid_argument, "the copy of an nlopt_opt object failed");
        }
    }
    nlopt_obj(nlopt_obj &&) = default;
    nlopt_obj &operator=(const nlopt_obj &) = delete;
    nlopt_obj &operator=(nlopt_obj &&) = delete;
    std::string get_name() const
    {
        return ::nlopt_algorithm_name(::nlopt_get_algorithm(m_value.get()));
    }
    problem &m_prob;
    sparsity_pattern m_sp;
    std::unique_ptr<std::remove_pointer<::nlopt_opt>::type, void (*)(::nlopt_opt)> m_value;
    vector_double m_dv;
};
}
}

#endif
