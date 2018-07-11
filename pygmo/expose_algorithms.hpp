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

#ifndef PYGMO_EXPOSE_ALGORITHMS_HPP
#define PYGMO_EXPOSE_ALGORITHMS_HPP

#include <pygmo/python_includes.hpp>

#include <string>

#include <boost/any.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/init.hpp>
#include <boost/python/object.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/str.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/docstrings.hpp>
#include <pygmo/pygmo_classes.hpp>

namespace pygmo
{

// Split algorithm exposition functions.
void expose_algorithms_0();
void expose_algorithms_1();

// A couple of utilities useful in the implementation
// of expose_algorithms_n().
namespace bp = boost::python;

template <typename T>
inline void expose_not_population_based(bp::class_<T> &c, const std::string &algo_name)
{
    using namespace pagmo;
    // Selection/replacement.
    add_property(
        c, "selection", lcast([](const T &n) -> bp::object {
            auto s = n.get_selection();
            if (boost::any_cast<std::string>(&s)) {
                return bp::str(boost::any_cast<std::string>(s));
            }
            return bp::object(boost::any_cast<population::size_type>(s));
        }),
        lcast([](T &n, const bp::object &o) {
            bp::extract<std::string> e_str(o);
            if (e_str.check()) {
                n.set_selection(e_str());
                return;
            }
            bp::extract<population::size_type> e_idx(o);
            if (e_idx.check()) {
                n.set_selection(e_idx());
                return;
            }
            pygmo_throw(::PyExc_TypeError,
                        ("cannot convert the input object '" + str(o) + "' of type '" + str(type(o))
                         + "' to either a selection policy (one of ['best', 'worst', 'random']) or an individual index")
                            .c_str());
        }),
        bls_selection_docstring(algo_name).c_str());
    add_property(
        c, "replacement", lcast([](const T &n) -> bp::object {
            auto s = n.get_replacement();
            if (boost::any_cast<std::string>(&s)) {
                return bp::str(boost::any_cast<std::string>(s));
            }
            return bp::object(boost::any_cast<population::size_type>(s));
        }),
        lcast([](T &n, const bp::object &o) {
            bp::extract<std::string> e_str(o);
            if (e_str.check()) {
                n.set_replacement(e_str());
                return;
            }
            bp::extract<population::size_type> e_idx(o);
            if (e_idx.check()) {
                n.set_replacement(e_idx());
                return;
            }
            pygmo_throw(
                ::PyExc_TypeError,
                ("cannot convert the input object '" + str(o) + "' of type '" + str(type(o))
                 + "' to either a replacement policy (one of ['best', 'worst', 'random']) or an individual index")
                    .c_str());
        }),
        bls_replacement_docstring(algo_name).c_str());
    c.def("set_random_sr_seed", &T::set_random_sr_seed, bls_set_random_sr_seed_docstring(algo_name).c_str());
}

// Main algorithm exposition function - for *internal* use by pygmo. The exposition function
// for APs needs to be different.
template <typename Algo>
inline bp::class_<Algo> expose_algorithm_pygmo(const char *name, const char *descr)
{
    // We require all algorithms to be def-ctible at the bare minimum.
    bp::class_<Algo> c(name, descr, bp::init<>());

    // Mark it as a C++ algorithm.
    c.attr("_pygmo_cpp_algorithm") = true;

    // Get reference to the algorithm class.
    auto &algo = get_algorithm_class();

    // Expose the algorithm constructor from Algo.
    algo.def(bp::init<const Algo &>((bp::arg("uda"))));

    // Expose extract.
    algo.def("_cpp_extract", &generic_cpp_extract<pagmo::algorithm, Algo>, bp::return_internal_reference<>());

    // Add the algorithm to the algorithms submodule.
    bp::scope().attr("algorithms").attr(name) = c;

    return c;
}
} // namespace pygmo

#endif
