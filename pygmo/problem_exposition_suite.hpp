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

#ifndef PYGMO_PROBLEM_EXPOSITION_SUITE_HPP
#define PYGMO_PROBLEM_EXPOSITION_SUITE_HPP

#include "python_includes.hpp"

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/init.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/object.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>
#include <cassert>
#include <memory>
#include <string>
#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"
#include "pygmo_classes.hpp"

namespace pygmo
{

namespace bp = boost::python;

// Wrapper for the best known method.
// NOTE: abstracted here because it is used in multiple places.
template <typename Prob>
inline bp::object best_known_wrapper(const Prob &p)
{
    return v_to_a(p.best_known());
}

// Expose a problem ctor from a C++ UDP.
// NOTE: abstracted in a separate wrapper because it is re-used in core.cpp.
template <typename Prob>
inline void problem_expose_init_cpp_udp()
{
    assert(problem_ptr.get() != nullptr);
    auto &prob_class = *problem_ptr;
    prob_class.def(bp::init<const Prob &>((bp::arg("udp"))));
}

// Main C++ UDP exposition function.
template <typename Prob>
inline bp::class_<Prob> expose_problem(const char *name, const char *descr)
{
    assert(problem_ptr.get() != nullptr);
    auto &problem_class = *problem_ptr;
    // We require all problems to be def-ctible at the bare minimum.
    bp::class_<Prob> c(name, descr, bp::init<>());
    // Mark it as a C++ problem.
    c.attr("_pygmo_cpp_problem") = true;

    // Expose the problem constructor from Prob.
    problem_expose_init_cpp_udp<Prob>();
    // Expose extract.
    problem_class.def("_cpp_extract", &generic_cpp_extract<pagmo::problem, Prob>, bp::return_internal_reference<>());

    // Add the problem to the problems submodule.
    bp::scope().attr("problems").attr(name) = c;

    return c;
}
}

#endif
