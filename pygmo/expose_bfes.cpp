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

#include <pagmo/batch_evaluators/default_bfe.hpp>
#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include <pygmo/docstrings.hpp>
#include <pygmo/expose_bfes.hpp>

using namespace pagmo;

namespace pygmo
{

namespace detail
{

namespace
{

// A test bfe.
struct test_bfe {
    vector_double operator()(const problem &p, const vector_double &dvs) const
    {
        vector_double retval;
        const auto nx = p.get_nx();
        const auto n_dvs = dvs.size() / nx;
        for (decltype(dvs.size()) i = 0; i < n_dvs; ++i) {
            const auto f = p.fitness(vector_double(dvs.data() + i * nx, dvs.data() + (i + 1u) * nx));
            retval.insert(retval.end(), f.begin(), f.end());
        }
        return retval;
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

void expose_bfes()
{
    // Test bfe.
    auto t_bfe = expose_bfe_pygmo<detail::test_bfe>("_test_bfe", "A test bfe.");
    t_bfe.def("get_n", &detail::test_bfe::get_n);
    t_bfe.def("set_n", &detail::test_bfe::set_n);

    // Default bfe.
    expose_bfe_pygmo<default_bfe>("default_bfe", default_bfe_docstring().c_str());

    // Thread bfe.
    expose_bfe_pygmo<thread_bfe>("thread_bfe", thread_bfe_docstring().c_str());

    // Member bfe.
    expose_bfe_pygmo<member_bfe>("member_bfe", member_bfe_docstring().c_str());
}

} // namespace pygmo
