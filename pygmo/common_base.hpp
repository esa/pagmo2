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

#ifndef PYGMO_COMMON_BASE_HPP
#define PYGMO_COMMON_BASE_HPP

#include "python_includes.hpp"

#include <boost/python/extract.hpp>
#include <boost/python/object.hpp>
#include <string>

#include "common_utils.hpp"

namespace pygmo
{

namespace bp = boost::python;

// A common base class with methods useful inthe implementation of
// the pythonic problem and algorithm.
struct common_base {
    // Try to get an attribute from an object. If the call fails,
    // return a def-cted object.
    static bp::object try_attr(const bp::object &o, const char *s)
    {
        bp::object a;
        try {
            a = o.attr(s);
        } catch (...) {
            PyErr_Clear();
        }
        return a;
    }
    // Throw if object does not have a callable attribute.
    static void check_callable_attribute(const bp::object &o, const char *s, const char *target)
    {
        bp::object a;
        try {
            a = o.attr(s);
        } catch (...) {
            pygmo_throw(PyExc_TypeError, ("the mandatory '" + std::string(s) + "()' method is missing from the "
                                                                               "user-defined Python "
                                          + std::string(target) + " '" + str(o) + "' of type '" + str(type(o)) + "'")
                                             .c_str());
        }
        if (!pygmo::callable(a)) {
            pygmo_throw(PyExc_TypeError,
                        ("the mandatory '" + std::string(s) + "()' method in the "
                                                              "user-defined Python "
                         + std::string(target) + " '" + str(o) + "' of type '" + str(type(o)) + "' is "
                                                                                                "not callable")
                            .c_str());
        }
    }
    // A simple wrapper for getters.
    template <typename RetType>
    static RetType getter_wrapper(const bp::object &o, const char *name, const RetType &def_value)
    {
        auto a = try_attr(o, name);
        if (a) {
            return bp::extract<RetType>(a());
        }
        return def_value;
    }
};
}

#endif
