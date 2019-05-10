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

#ifndef PYGMO_COMMON_BASE_HPP
#define PYGMO_COMMON_BASE_HPP

#include <pygmo/python_includes.hpp>

#include <boost/python/extract.hpp>
#include <boost/python/object.hpp>

#include <pygmo/common_utils.hpp>

namespace pygmo
{

namespace bp = boost::python;

// A common base class with methods useful in the implementation of
// the pythonic problem, algorithm, etc.
struct common_base {
    static void check_mandatory_method(const bp::object &, const char *, const char *);
    // A simple wrapper for getters. It will try to:
    // - get the attribute "name" from the object o,
    // - call it without arguments,
    // - extract an instance from the ret value and return it.
    // If the attribute is not there or it is not callable, the value "def_value" will be returned.
    template <typename RetType>
    static RetType getter_wrapper(const bp::object &o, const char *name, const RetType &def_value)
    {
        auto a = callable_attribute(o, name);
        if (a.is_none()) {
            return def_value;
        }
        return bp::extract<RetType>(a());
    }
    static void check_not_type(const bp::object &, const char *);
};
} // namespace pygmo

#endif
