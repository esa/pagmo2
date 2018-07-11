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

#ifndef PYGMO_EXPOSE_PROBLEMS_HPP
#define PYGMO_EXPOSE_PROBLEMS_HPP

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>

#include <pagmo/problem.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/pygmo_classes.hpp>

namespace pygmo
{

// Split problem exposition functions.
void expose_problems_0();
void expose_problems_1();

// A couple of utilities useful in the implementation
// of expose_problems_n().
namespace bp = boost::python;

// C++ UDP exposition function - for *internal* pygmo use. This needs to be different
// from the exposition function used for APs.
template <typename Prob>
inline bp::class_<Prob> expose_problem_pygmo(const char *name, const char *descr)
{
    using namespace pagmo;

    // We require all problems to be def-ctible at the bare minimum.
    bp::class_<Prob> c(name, descr, bp::init<>());

    // Mark it as a C++ problem.
    c.attr("_pygmo_cpp_problem") = true;

    // Get reference to the problem class.
    auto &prob = get_problem_class();

    // Expose the problem constructor from Prob.
    prob.def(bp::init<const Prob &>((bp::arg("udp"))));

    // Expose extract.
    prob.def("_cpp_extract", &generic_cpp_extract<pagmo::problem, Prob>, bp::return_internal_reference<>());

    // Add the problem to the problems submodule.
    bp::scope().attr("problems").attr(name) = c;

    return c;
}
} // namespace pygmo

#endif
