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

#ifndef PYGMO_ALGORITHM_EXPOSITION_SUITE_HPP
#define PYGMO_ALGORITHM_EXPOSITION_SUITE_HPP

#include "python_includes.hpp"

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/list.hpp>
#include <boost/python/scope.hpp>
#include <cassert>

#include "common_utils.hpp"
#include "pygmo_classes.hpp"

namespace pygmo
{

namespace bp = boost::python;

// Expose an algorithm ctor from a user-defined algorithm.
template <typename Algo>
inline void algorithm_algo_init()
{
    assert(algorithm_ptr.get() != nullptr);
    auto &algo_class = *algorithm_ptr;
    algo_class.def(bp::init<const Algo &>((bp::arg("a"))));
}

// Utils to expose algo log.
template <typename Algo>
inline bp::list generic_log_getter(const Algo &a)
{
    bp::list retval;
    for (const auto &t : a.get_log()) {
        retval.append(cpptuple_to_pytuple(t));
    }
    return retval;
}

template <typename Algo>
inline void expose_algo_log(bp::class_<Algo> &algo_class, const char *doc)
{
    algo_class.def("get_log", &generic_log_getter<Algo>, doc);
}

// Main algorithm exposition function.
template <typename Algo>
inline bp::class_<Algo> expose_algorithm(const char *name, const char *descr)
{
    assert(algorithm_ptr.get() != nullptr);
    auto &algorithm_class = *algorithm_ptr;
    // We require all algorithms to be def-ctible at the bare minimum.
    bp::class_<Algo> c(name, descr, bp::init<>());
    // Mark it as a C++ algorithm.
    c.attr("_pygmo_cpp_algorithm") = true;

    // Expose the algorithm constructor from Algo.
    algorithm_algo_init<Algo>();
    // Expose extract.
    algorithm_class.def("_cpp_extract", &generic_cpp_extract<pagmo::algorithm, Algo>);

    // Add the algorithm to the algorithms submodule.
    bp::scope().attr("algorithms").attr(name) = c;

    return c;
}
}

#endif
