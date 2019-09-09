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

#ifndef PYGMO_PROBLEM_EXPOSITION_SUITE_HPP
#define PYGMO_PROBLEM_EXPOSITION_SUITE_HPP

#include <pygmo/python_includes.hpp>

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/import.hpp>
#include <boost/python/init.hpp>
#include <boost/python/object.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <cstdint>
#include <memory>

#include <pagmo/problem.hpp>

#include <pygmo/common_utils.hpp>

namespace pygmo
{

namespace bp = boost::python;

// Main C++ UDP exposition function for use by APs.
template <typename Prob>
inline bp::class_<Prob> expose_problem(const char *name, const char *descr)
{
    // We require all problems to be def-ctible at the bare minimum.
    bp::class_<Prob> c(name, descr, bp::init<>());

    // Mark it as a C++ problem.
    c.attr("_pygmo_cpp_problem") = true;

    // Get the problem class from the pygmo module.
    auto &prob = **reinterpret_cast<std::unique_ptr<bp::class_<pagmo::problem>> *>(
        bp::extract<std::uintptr_t>(bp::import("pygmo").attr("core").attr("_problem_address"))());

    // Expose the problem constructor from Prob.
    prob.def(bp::init<const Prob &>((bp::arg("udp"))));

    // Expose extract.
    prob.def("_cpp_extract", &generic_cpp_extract<pagmo::problem, Prob>, bp::return_internal_reference<>());

    return c;
}

// Wrapper for the best known method.
template <typename Prob>
inline bp::object best_known_wrapper(const Prob &p)
{
    return vector_to_ndarr(p.best_known());
}
} // namespace pygmo

#endif
