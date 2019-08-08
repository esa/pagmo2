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

// Disable various warnings from MSVC.
#pragma warning(disable : 4275)
#pragma warning(disable : 4996)
#pragma warning(disable : 4503)
#pragma warning(disable : 4244)

#endif

#include <pygmo/python_includes.hpp>

// See: https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// In every cpp file we need to make sure this is included before everything else,
// with the correct #defines.
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygmo_ARRAY_API
#include <pygmo/numpy.hpp>

#include <pagmo/s_policies/select_best.hpp>
#include <pagmo/types.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/docstrings.hpp>
#include <pygmo/expose_s_policies.hpp>
#include <pygmo/sr_policy_add_rate_constructor.hpp>

using namespace pagmo;
namespace bp = boost::python;

namespace pygmo
{

namespace detail
{

namespace
{

// A test s_policy.
struct test_s_policy {
    individuals_group_t select(const individuals_group_t &inds, const vector_double::size_type &,
                               const vector_double::size_type &, const vector_double::size_type &,
                               const vector_double::size_type &, const vector_double::size_type &,
                               const vector_double &) const
    {
        return inds;
    }
    // Set/get an internal value to test extraction semantics.
    void set_n(int n)
    {
        m_n = n;
    }
    int get_n() const
    {
        return m_n;
    }
    int m_n = 1;
};

} // namespace

} // namespace detail

void expose_s_policies()
{
    // Test s_policy.
    auto t_s_policy = expose_s_policy_pygmo<detail::test_s_policy>("_test_s_policy", "A test selection policy.");
    t_s_policy.def("get_n", &detail::test_s_policy::get_n);
    t_s_policy.def("set_n", &detail::test_s_policy::set_n);

    // Select best policy.
    auto select_best_ = expose_s_policy_pygmo<select_best>("select_best", select_best_docstring().c_str());
    detail::sr_policy_add_rate_constructor(select_best_);
}

} // namespace pygmo
