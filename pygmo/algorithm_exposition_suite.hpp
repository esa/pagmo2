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

#ifndef PYGMO_ALGORITHM_EXPOSITION_SUITE_HPP
#define PYGMO_ALGORITHM_EXPOSITION_SUITE_HPP

#include <pygmo/python_includes.hpp>

#include <cstdint>
#include <memory>

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/import.hpp>
#include <boost/python/init.hpp>
#include <boost/python/list.hpp>
#include <boost/python/return_internal_reference.hpp>

#include <pagmo/algorithm.hpp>

#include <pygmo/common_utils.hpp>

namespace pygmo
{

namespace bp = boost::python;

// Main algorithm exposition function for use by APs.
template <typename Algo>
inline bp::class_<Algo> expose_algorithm(const char *name, const char *descr)
{
    // We require all algorithms to be def-ctible at the bare minimum.
    bp::class_<Algo> c(name, descr, bp::init<>());

    // Mark it as a C++ algorithm.
    c.attr("_pygmo_cpp_algorithm") = true;

    // Get the algorithm class from the pygmo module.
    auto &algo = **reinterpret_cast<std::unique_ptr<bp::class_<pagmo::algorithm>> *>(
        bp::extract<std::uintptr_t>(bp::import("pygmo").attr("core").attr("_algorithm_address"))());

    // Expose the algorithm constructor from Algo.
    algo.def(bp::init<const Algo &>((bp::arg("uda"))));

    // Expose extract.
    algo.def("_cpp_extract", &generic_cpp_extract<pagmo::algorithm, Algo>, bp::return_internal_reference<>());

    return c;
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
} // namespace pygmo
#endif
