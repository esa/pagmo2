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

#include <pygmo/python_includes.hpp>

// See: https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// In every cpp file we need to make sure this is included before everything else,
// with the correct #defines.
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygmo_ARRAY_API
#include <pygmo/numpy.hpp>

#include <cassert>
#include <stdexcept>
#include <string>

#include <boost/python/errors.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/handle.hpp>
#include <boost/python/import.hpp>
#include <boost/python/str.hpp>

#include <pygmo/handle_thread_py_exception.hpp>

namespace pygmo
{

// Helper to handle Python exceptions thrown in a separate thread of execution not managed
// by Python. In such cases we want to capture the exception, reset the Python error flag,
// and repackage the error message as a C++ exception.
void handle_thread_py_exception(const std::string &err)
{
    namespace bp = boost::python;

    // NOTE: my understanding is that this assert should never fail, if we are handling a bp::error_already_set
    // exception it means a Python exception was generated. However, I have seen snippets of code on the
    // internet where people do check this flag. Keep this in mind, it should be easy to transform this assert()
    // in an if/else.
    assert(::PyErr_Occurred());

    // Small helper to build a bp::object from a raw PyObject ptr.
    // It assumes that ptr is a new reference, or null. If null, we
    // return None.
    auto new_ptr_to_obj = [](::PyObject *ptr) { return ptr ? bp::object(bp::handle<>(ptr)) : bp::object(); };

    // Fetch the error data that was set by Python: exception type, value and the traceback.
    ::PyObject *type, *value, *traceback;
    // PyErr_Fetch() creates new references, and it also clears the error indicator.
    ::PyErr_Fetch(&type, &value, &traceback);
    assert(!::PyErr_Occurred());
    // This normalisation step is apparently needed because sometimes, for some Python-internal reasons,
    // the values returned by PyErr_Fetch() are “unnormalized” (see the Python documentation for this function).
    ::PyErr_NormalizeException(&type, &value, &traceback);
    // Move them into bp::object, so that they are cleaned up at the end of the scope. These are all new
    // objects.
    auto tp = new_ptr_to_obj(type);
    auto v = new_ptr_to_obj(value);
    auto tb = new_ptr_to_obj(traceback);

    // Try to extract a string description of the exception using the "traceback" module.
    std::string tmp(err);
    try {
        // NOTE: we are about to go back into the Python interpreter. Here Python could throw an exception
        // and set again the error indicator, which was reset above by PyErr_Fetch(). In case of any issue,
        // we will give up any attempt of producing a meaningful error message, reset the error indicator,
        // and throw a pure C++ exception with a generic error message.
        tmp += bp::extract<std::string>(
            bp::str("").attr("join")(bp::import("traceback").attr("format_exception")(tp, v, tb)));
    } catch (const bp::error_already_set &) {
        // The block above threw from Python. There's not much we can do.
        ::PyErr_Clear();
        throw std::runtime_error("While trying to analyze the error message of a Python exception raised in a "
                                 "separate thread, another Python exception was raised. Giving up now.");
    }
    // Throw the C++ exception.
    throw std::runtime_error(tmp);
}

} // namespace pygmo
