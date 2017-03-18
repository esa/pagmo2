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

#ifndef PYGMO_ISLAND_HPP
#define PYGMO_ISLAND_HPP

#include "python_includes.hpp"

#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/class.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/handle.hpp>
#include <boost/python/object.hpp>
#include <boost/python/str.hpp>
#include <boost/python/tuple.hpp>
#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>

#include "common_base.hpp"
#include "common_utils.hpp"

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

// Disable the static UDI checks for bp::object.
template <>
struct disable_udi_checks<bp::object> : std::true_type {
};

template <>
struct isl_inner<bp::object> final : isl_inner_base, pygmo::common_base {
    template <typename T>
    static void check_thread_safety(const T &x)
    {
        if (static_cast<int>(x.get_thread_safety()) < static_cast<int>(thread_safety::copyonly)) {
            pygmo_throw(PyExc_ValueError,
                        ("pythonic islands require objects which provide at least the copyonly thread "
                         "safety level, but the object '"
                         + x.get_name() + "' does not provide any thread safety guarantee")
                            .c_str());
        }
    }
    // Just need the def ctor, delete everything else.
    isl_inner() = default;
    isl_inner(const isl_inner &) = delete;
    isl_inner(isl_inner &&) = delete;
    isl_inner &operator=(const isl_inner &) = delete;
    isl_inner &operator=(isl_inner &&) = delete;
    explicit isl_inner(const bp::object &o)
    {
        check_not_type(o, "island");
        check_mandatory_method(o, "run_evolve", "island");
        m_value = pygmo::deepcopy(o);
    }
    virtual std::unique_ptr<isl_inner_base> clone() const override final
    {
        // This will make a deep copy using the ctor above.
        return make_unique<isl_inner>(m_value);
    }
    // Mandatory methods.
    virtual void run_evolve(algorithm &algo, std::mutex &algo_mutex, population &pop,
                            std::mutex &pop_mutex) const override final
    {
        // NOTE: run_evolve() is called from a separate thread in pagmo::island, need to construct a GTE before
        // doing anything with the interpreter (including the throws in the checks below).
        // NOTE: we must make sure that we lock the GIL before locking algo and pop. The reason is that in other
        // situations the GIL is always the first lock to be locked (e.g., if we do a get_population(), get_algo(),
        // etc.). In this situation, we are calling this code from a separate C++ thread which, before the
        // construction of the ensurer, does NOT hold the GIL. If we locked the algo/pop before the GIL, we would end
        // up with a lock order inversion with potential deadlock.
        pygmo::gil_thread_ensurer gte;
        std::unique_lock<std::mutex> algo_lock(algo_mutex), pop_lock(pop_mutex);
        try {
            // NOTE: the idea of these checks is the following: we will have to copy algo and pop in order to invoke
            // the pythonic UDI's evolve method, which has a signature which is different from C++. If algo/prob are
            // bp::object, via the GTE above we have made sure we can safely copy them so we don't need to check
            // anything. Otherwise, we run the check, which is against the copyonly thread safety level as that is all
            // we need.
            if (!algo.is<bp::object>()) {
                check_thread_safety(algo);
            }
            if (!pop.get_problem().is<bp::object>()) {
                check_thread_safety(pop.get_problem());
            }

            // Everything fine, copy algo/pop and unlock.
            bp::object pop_copy(pop);
            pop_lock.unlock();
            bp::object algo_copy(algo);
            algo_lock.unlock();

            // Invoke the run_evolve() method of the UDI and get out the evolved pop.
            // NOTE: here bp::extract will extract a copy of pop, and then a move ctor will take place,
            // which will just move the internal problem pointer. No additional thread safety guarantees are needed.
            population new_pop = bp::extract<population>(m_value.attr("run_evolve")(algo_copy, pop_copy));
            // Re-lock and assign.
            pop_lock.lock();
            pop = std::move(new_pop);
        } catch (const bp::error_already_set &) {
            // NOTE: run_evolve() is called from a separate thread. If Python raises any exception in this separate
            // thread (as signalled by the bp::error_already_set exception being handled here), the following will
            // happen: the Python error indicator has been set for the *current* thread, but the bp::error_already_set
            // exception will actually *escape* this thread due to the internal exception transport mechanism of
            // std::future. In other words, bp::error_already_set will be re-thrown in a thread which, from the
            // Python side, has no knowledge/information about the Python exception that originated all this, resulting
            // in an unhelpful error message by Boost Python.
            //
            // What we do then is the following: we get the Python exception via PyErr_Fetch(), store its data in an
            // ad-hoc C++ exception, which will be thrown and then transferred by std::future to the thread that calls
            // wait() on the future.
            // https://docs.python.org/3/c-api/exceptions.html
            //
            // NOTE: we used to have here a more sophisticated system that attempted to transport the exception
            // information for Python into a C++ exception, that would then be translated back to a Python exception via
            // a translator registered in core.cpp. However, managing the lifetime of the exception data turned out to
            // be difficult, with either segfaults in certain situations or memory leaks (if the pointers are never
            // freed). Maybe we can revisit this at one point in the future. The relevant code can be found at the git
            // revision 13a2d254a62dee5d82858595f95babd145e91e94.
            //
            // NOTE: my understanding is that this assert should never fail, if we are handling a bp::error_already_set
            // exception it means a Python exception was generated. However, I have seen snippets of code on the
            // internet where people do check this flag. Keep this in mind, it should be easy to transform this assert()
            // in an if/else.
            assert(::PyErr_Occurred());
            // Fetch the error data that was set by Python: exception type, value and the traceback.
            ::PyObject *type, *value, *traceback;
            // PyErr_Fetch() creates new references, and it also clears the error indicator.
            ::PyErr_Fetch(&type, &value, &traceback);
            // This normalisation step is apparently needed because sometimes, for some Python-internal reasons,
            // the values returned by PyErr_Fetch() are “unnormalized” (see the Python documentation for this function).
            ::PyErr_NormalizeException(&type, &value, &traceback);
            // Move them into bp::object, so that they are cleaned up at the end of the scope. These are all new
            // objects.
            bp::object tp{bp::handle<>{type}}, v{bp::handle<>{value}}, tb{bp::handle<>{traceback}};
            // Extract a string description of the exception using the "traceback" module.
            const std::string tmp = bp::extract<std::string>(
                bp::str("").attr("join")(bp::import("traceback").attr("format_exception")(tp, v, tb)));
            // Throw the C++ exception.
            throw std::runtime_error("The asynchronous evolution of a Pythonic island of type '" + get_name()
                                     + "' raised an error:\n" + tmp);
        }
    }
    // Optional methods.
    virtual std::string get_name() const override final
    {
        return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
    }
    virtual std::string get_extra_info() const override final
    {
        return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
    }
    bp::object m_value;
};
}
}

// Register the isl_inner specialisation for bp::object.
PAGMO_REGISTER_ISLAND(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

// Serialization support for the island class.
struct island_pickle_suite : bp::pickle_suite {
    static bp::tuple getstate(const pagmo::island &isl)
    {
        // The idea here is that first we extract a char array
        // into which island has been cerealised, then we turn
        // this object into a Python bytes object and return that.
        std::ostringstream oss;
        {
            cereal::PortableBinaryOutputArchive oarchive(oss);
            oarchive(isl);
        }
        auto s = oss.str();
        return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())));
    }
    static void setstate(pagmo::island &isl, const bp::tuple &state)
    {
        // Similarly, first we extract a bytes object from the Python state,
        // and then we build a C++ string from it. The string is then used
        // to decerealise the object.
        if (len(state) != 1) {
            pygmo_throw(PyExc_ValueError, ("the state tuple passed for island deserialization "
                                           "must have a single element, but instead it has "
                                           + std::to_string(len(state)) + " elements")
                                              .c_str());
        }
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
            cereal::PortableBinaryInputArchive iarchive(iss);
            iarchive(isl);
        }
    }
};
}

#endif
