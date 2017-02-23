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
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>
#include <cassert>
#include <pagmo/algorithms/mbh.hpp>
#include <pagmo/rng.hpp>

#include "common_utils.hpp"
#include "pygmo_classes.hpp"

namespace pygmo
{

namespace bp = boost::python;

// Machinery for the exposition of mbh constructors.

// Abstract this away as it is shared in the two following functions.
template <typename Algo>
inline pagmo::mbh *mbh_init_impl(const Algo &a, unsigned stop, const bp::object &perturb, unsigned seed)
{
    // NOTE: the idea here is that if perturb is a float (or derives from it),
    // we will select the C++ ctor with scalar perturb. Otherwise, we will attempt
    // to invoke the ctor with vector perturb.
    const auto float_tp = builtin().attr("float");
    if (isinstance(perturb, float_tp)) {
        const double p = bp::extract<double>(perturb);
        return ::new pagmo::mbh(a, stop, p, seed);
    }
    const auto p = to_vd(perturb);
    return ::new pagmo::mbh(a, stop, p, seed);
}

// Constructor from C++ UDA.
template <typename Algo>
inline pagmo::mbh *mbh_init(const Algo &a, unsigned stop, const bp::object &perturb, unsigned seed)
{
    return mbh_init_impl(a, stop, perturb, seed);
}

// Constructor from Python UDA.
template <>
inline pagmo::mbh *mbh_init<bp::object>(const bp::object &a, unsigned stop, const bp::object &perturb, unsigned seed)
{
    // NOTE: as usual, when constructing from a generic Python UDA we must prevent construction from
    // pagmo::algorithm, in order to match the C++ behaviour.
    if (type(a) == *algorithm_ptr) {
        pygmo_throw(PyExc_TypeError, "a pygmo algorithm is not a user-defined algorithm, and it cannot be used "
                                     "as a construction argument for the mbh meta-algorithm");
    }
    return mbh_init_impl(a, stop, perturb, seed);
}

// Make python inits for mbh from the above ctors.
template <typename Algo>
inline void make_mbh_inits(bp::class_<pagmo::mbh> &mbh_)
{
    // We have two inits: one with explicit seed, the other with default seed (generated randomly).
    mbh_.def("__init__", bp::make_constructor(+[](const Algo &a, unsigned stop, const bp::object &perturb,
                                                  unsigned seed) { return mbh_init(a, stop, perturb, seed); },
                                              bp::default_call_policies(),
                                              (bp::arg("uda"), bp::arg("stop"), bp::arg("perturb"), bp::arg("seed"))));
    mbh_.def("__init__", bp::make_constructor(
                             +[](const Algo &a, unsigned stop, const bp::object &perturb) {
                                 return mbh_init(a, stop, perturb, pagmo::random_device::next());
                             },
                             bp::default_call_policies(), (bp::arg("uda"), bp::arg("stop"), bp::arg("perturb"))));
}

// Expose an algorithm ctor from a C++ UDA.
template <typename Algo>
inline void algorithm_expose_init_cpp_uda()
{
    assert(algorithm_ptr.get() != nullptr);
    auto &algo_class = *algorithm_ptr;
    algo_class.def(bp::init<const Algo &>((bp::arg("uda"))));
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
    assert(mbh_ptr.get() != nullptr);
    auto &algorithm_class = *algorithm_ptr;
    auto &mbh_class = *mbh_ptr;
    // We require all algorithms to be def-ctible at the bare minimum.
    bp::class_<Algo> c(name, descr, bp::init<>());
    // Mark it as a C++ algorithm.
    c.attr("_pygmo_cpp_algorithm") = true;

    // Expose the algorithm constructor from Algo.
    algorithm_expose_init_cpp_uda<Algo>();
    // Expose extract.
    algorithm_class.def("_cpp_extract", &generic_cpp_extract<pagmo::algorithm, Algo>,
                        bp::return_internal_reference<>());

    // Expose mbh's constructors from Algo.
    make_mbh_inits<Algo>(mbh_class);
    // Extract.
    mbh_class.def("_cpp_extract", &generic_cpp_extract<pagmo::mbh, Algo>, bp::return_internal_reference<>());

    // Add the algorithm to the algorithms submodule.
    bp::scope().attr("algorithms").attr(name) = c;

    return c;
}
}

#endif
