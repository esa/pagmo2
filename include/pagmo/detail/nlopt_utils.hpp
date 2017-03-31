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
        // Init the nlopt_obj.
        if (!m_value) {
            pagmo_throw(std::invalid_argument, "the creation of an nlopt_opt object failed");
        }
        if (prob.get_nobj() != 1u) {
            // TODO
            pagmo_throw(std::invalid_argument, "" + get_name());
        }
        if (prob.get_nc()) {
            // TODO
            pagmo_throw(std::invalid_argument, "" + get_name());
        }
        m_dv.resize(prob.get_nx());
        // Set the objfun + gradient.
        auto res = ::nlopt_set_min_objective(m_value.get(),
                                             [](unsigned dim, const double *x, double *grad, void *f_data) {
                                                 auto &nlo = *static_cast<nlopt_obj *>(f_data);
                                                 auto &p = nlo.m_prob;
                                                 auto &dv = nlo.m_dv;
                                                 auto &sp = nlo.m_sp;
                                                 assert(dim == p.get_nx());
                                                 if (grad && !p.has_gradient()) {
                                                     // TODO
                                                     pagmo_throw(std::invalid_argument, "" + nlo.get_name());
                                                 }
                                                 assert(dv.size() == dim);
                                                 std::copy(x, x + dim, dv.begin());
                                                 const auto fitness = p.fitness(dv);
                                                 const auto gradient = p.gradient(dv);
                                                 auto g_it = gradient.begin();
                                                 const auto g_end = gradient.end();
                                                 auto i = 0u;
                                                 for (auto sp_it = sp.begin(); i < dim && g_it != g_end;
                                                      ++i, ++g_it, ++sp_it) {
                                                     assert(sp_it->first == 0u);
                                                 }
                                                 return fitness[0];
                                             },
                                             static_cast<void *>(this));
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
