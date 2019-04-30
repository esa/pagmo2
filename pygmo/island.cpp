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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/handle.hpp>
#include <boost/python/import.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/str.hpp>
#include <boost/python/tuple.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/island.hpp>

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

isl_inner<bp::object>::isl_inner(const bp::object &o)
{
    // NOTE: unlike pygmo.problem/algorithm, we don't have to enforce
    // that o is not a pygmo.island, because pygmo.island does not satisfy
    // the UDI interface and thus check_mandatory_method() below will fail
    // if o is a pygmo.island.
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "island");
    check_mandatory_method(o, "run_evolve", "island");
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<isl_inner_base> isl_inner<bp::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return detail::make_unique<isl_inner>(m_value);
}

void isl_inner<bp::object>::handle_thread_py_exception(const std::string &err)
{
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

void isl_inner<bp::object>::run_evolve(island &isl) const
{
    // NOTE: run_evolve() is called from a separate thread in pagmo::island, need to construct a GTE before
    // doing anything with the interpreter (including the throws in the checks below).
    pygmo::gil_thread_ensurer gte;

    // NOTE: every time we call into the Python interpreter from a separate thread, we need to
    // handle Python exceptions in a special way.
    std::string isl_name;
    try {
        isl_name = get_name();
    } catch (const bp::error_already_set &) {
        handle_thread_py_exception("Could not fetch the name of the pythonic island. The error is:\n");
    }

    try {
        auto ret = m_value.attr("run_evolve")(isl.get_algorithm(), isl.get_population());
        bp::extract<bp::tuple> ext_ret(ret);
        if (!ext_ret.check()) {
            pygmo_throw(PyExc_TypeError, ("the 'run_evolve()' method of a user-defined island "
                                          "must return a tuple, but it returned an object of type '"
                                          + pygmo::str(pygmo::type(ret)) + "' instead")
                                             .c_str());
        }
        bp::tuple ret_tup = ext_ret;
        if (len(ret_tup) != 2) {
            pygmo_throw(PyExc_ValueError, ("the tuple returned by the 'run_evolve()' method of a user-defined island "
                                           "must have 2 elements, but instead it has "
                                           + std::to_string(len(ret_tup)) + " element(s)")
                                              .c_str());
        }
        bp::extract<algorithm> ret_algo(ret_tup[0]);
        if (!ret_algo.check()) {
            pygmo_throw(PyExc_TypeError,
                        ("the first value returned by the 'run_evolve()' method of a user-defined island "
                         "must be an algorithm, but an object of type '"
                         + pygmo::str(pygmo::type(ret_tup[0])) + "' was returned instead")
                            .c_str());
        }
        bp::extract<population> ret_pop(ret_tup[1]);
        if (!ret_pop.check()) {
            pygmo_throw(PyExc_TypeError,
                        ("the second value returned by the 'run_evolve()' method of a user-defined island "
                         "must be a population, but an object of type '"
                         + pygmo::str(pygmo::type(ret_tup[1])) + "' was returned instead")
                            .c_str());
        }
        isl.set_algorithm(ret_algo);
        isl.set_population(ret_pop);
    } catch (const bp::error_already_set &) {
        handle_thread_py_exception("The asynchronous evolution of a Pythonic island of type '" + isl_name
                                   + "' raised an error:\n");
    }
}

std::string isl_inner<bp::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string isl_inner<bp::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_ISLAND_IMPLEMENT(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

bp::tuple island_pickle_suite::getstate(const pagmo::island &isl)
{
    // The idea here is that first we extract a char array
    // into which island has been serialized, then we turn
    // this object into a Python bytes object and return that.
    std::ostringstream oss;
    {
        boost::archive::text_oarchive oarchive(oss);
        oarchive << isl;
    }
    auto s = oss.str();
    return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())), get_ap_list());
}

void island_pickle_suite::setstate(pagmo::island &isl, const bp::tuple &state)
{
    // Similarly, first we extract a bytes object from the Python state,
    // and then we build a C++ string from it. The string is then used
    // to deserialize the object.
    if (len(state) != 2) {
        pygmo_throw(PyExc_ValueError, ("the state tuple passed for island deserialization "
                                       "must have 2 elements, but instead it has "
                                       + std::to_string(len(state)) + " elements")
                                          .c_str());
    }

    // Make sure we import all the aps specified in the archive.
    import_aps(bp::list(state[1]));

    // NOTE: PyBytes_AsString is a macro.
    auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
    if (!ptr) {
        pygmo_throw(PyExc_TypeError, "a bytes object is needed to deserialize an island");
    }
    const auto size = len(state[0]);
    std::string s(ptr, ptr + size);
    std::istringstream iss;
    iss.str(s);
    {
        boost::archive::text_iarchive iarchive(iss);
        iarchive >> isl;
    }
}

} // namespace pygmo