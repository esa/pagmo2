/* Copyright 2017-2020 PaGMO development team

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
#include <string>

#include <boost/any.hpp>
#include <boost/python/class.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/import.hpp>
#include <boost/python/init.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/str.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>

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

// Utilities for implementing the exposition of algorithms
// which inherit from not_population_based.
inline std::string bls_selection_docstring(const std::string &algo)
{
    return R"(Individual selection policy.

This attribute represents the policy that is used in the ``evolve()`` method to select the individual
that will be optimised. The attribute can be either a string or an integral.

If the attribute is a string, it must be one of ``"best"``, ``"worst"`` and ``"random"``:

* ``"best"`` will select the best individual in the population,
* ``"worst"`` will select the worst individual in the population,
* ``"random"`` will randomly choose one individual in the population.

:func:`~pygmo.)"
           + algo + R"(.set_random_sr_seed()` can be used to seed the random number generator
used by the ``"random"`` policy.

If the attribute is an integer, it represents the index (in the population) of the individual that is selected
for optimisation.

Returns:
    ``int`` or ``str``: the individual selection policy or index

Raises:
    OverflowError: if the attribute is set to an integer which is negative or too large
    ValueError: if the attribute is set to an invalid string
    TypeError: if the attribute is set to a value of an invalid type
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

inline std::string bls_replacement_docstring(const std::string &algo)
{
    return R"(Individual replacement policy.

This attribute represents the policy that is used in the ``evolve()`` method to select the individual
that will be replaced by the optimised individual. The attribute can be either a string or an integral.

If the attribute is a string, it must be one of ``"best"``, ``"worst"`` and ``"random"``:

* ``"best"`` will select the best individual in the population,
* ``"worst"`` will select the worst individual in the population,
* ``"random"`` will randomly choose one individual in the population.

:func:`~pygmo.)"
           + algo + R"(.set_random_sr_seed()` can be used to seed the random number generator
used by the ``"random"`` policy.

If the attribute is an integer, it represents the index (in the population) of the individual that will be
replaced by the optimised individual.

Returns:
    ``int`` or ``str``: the individual replacement policy or index

Raises:
    OverflowError: if the attribute is set to an integer which is negative or too large
    ValueError: if the attribute is set to an invalid string
    TypeError: if the attribute is set to a value of an invalid type
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

inline std::string bls_set_random_sr_seed_docstring(const std::string &algo)
{
    return R"(set_random_sr_seed(seed)

Set the seed for the ``"random"`` selection/replacement policies.

Args:
    seed (``int``): the value that will be used to seed the random number generator used by the ``"random"``
      election/replacement policies (see :attr:`~pygmo.)"
           + algo + R"(.selection` and
      :attr:`~pygmo.)"
           + algo + R"(.replacement`)

Raises:
    OverflowError: if the attribute is set to an integer which is negative or too large
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

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

} // namespace pygmo
#endif
