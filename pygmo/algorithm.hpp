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

#ifndef PYGMO_ALGORITHM_HPP
#define PYGMO_ALGORITHM_HPP

#include "python_includes.hpp"

#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/class.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/object.hpp>
#include <boost/python/tuple.hpp>
#include <sstream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>
#include <pagmo/serialization.hpp>

#include "common_base.hpp"
#include "common_utils.hpp"
#include "object_serialization.hpp"

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

template <>
struct algo_inner<bp::object> final : algo_inner_base, pygmo::common_base {
    // These are the mandatory methods that must be present.
    void check_construction_object() const
    {
        if (pygmo::isinstance(m_value, pygmo::builtin().attr("type"))) {
            pygmo_throw(PyExc_TypeError, "cannot construct an algorithm from a type: please use an instance "
                                         "as construction argument");
        }
        check_mandatory_method(m_value, "evolve", "algorithm");
    }
    // Just need the def ctor, delete everything else.
    algo_inner() = default;
    algo_inner(const algo_inner &) = delete;
    algo_inner(algo_inner &&) = delete;
    algo_inner &operator=(const algo_inner &) = delete;
    algo_inner &operator=(algo_inner &&) = delete;
    explicit algo_inner(bp::object o)
        : // Perform an explicit deep copy of the input object.
          m_value(pygmo::deepcopy(o))
    {
        check_construction_object();
    }
    virtual algo_inner_base *clone() const override final
    {
        // This will make a deep copy using the ctor above.
        return ::new algo_inner(m_value);
    }
    // Mandatory methods.
    virtual population evolve(const population &pop) const override final
    {
        return bp::extract<population>(m_value.attr("evolve")(pop));
    }
    // Optional methods.
    virtual void set_seed(unsigned n) override final
    {
        auto a = pygmo::callable_attribute(m_value, "set_seed");
        if (a) {
            a(n);
        } else {
            pygmo_throw(PyExc_RuntimeError, "'set_seed()' has been called but it is not implemented");
        }
    }
    virtual bool has_set_seed() const override final
    {
        return getter_wrapper<bool>(m_value, "has_set_seed", pygmo::callable_attribute(m_value, "set_seed"));
    }
    virtual std::string get_name() const override final
    {
        return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
    }
    virtual std::string get_extra_info() const override final
    {
        return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
    }
    virtual void set_verbosity(unsigned n) override final
    {
        auto a = pygmo::callable_attribute(m_value, "set_verbosity");
        if (a) {
            a(n);
        } else {
            pygmo_throw(PyExc_RuntimeError, "'set_verbosity()' has been called but it is not implemented");
        }
    }
    virtual bool has_set_verbosity() const override final
    {
        return getter_wrapper<bool>(m_value, "has_set_verbosity", pygmo::callable_attribute(m_value, "set_verbosity"));
    }
    bp::object m_value;
};
}
}

// Register the algo_inner specialisation for bp::object.
PAGMO_REGISTER_ALGORITHM(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

// Serialization support for the algorithm class.
struct algorithm_pickle_suite : bp::pickle_suite {
    static bp::tuple getinitargs(const pagmo::algorithm &)
    {
        // For initialization purposes, we use the null algo.
        return bp::make_tuple(pagmo::null_algorithm{});
    }
    static bp::tuple getstate(const pagmo::algorithm &a)
    {
        // The idea here is that first we extract a char array
        // into which algorithm has been cerealised, then we turn
        // this object into a Python bytes object and return that.
        std::ostringstream oss;
        {
            cereal::PortableBinaryOutputArchive oarchive(oss);
            oarchive(a);
        }
        auto s = oss.str();
        return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())));
    }
    static void setstate(pagmo::algorithm &a, bp::tuple state)
    {
        // Similarly, first we extract a bytes object from the Python state,
        // and then we build a C++ string from it. The string is then used
        // to decerealise the object.
        if (len(state) != 1) {
            pygmo_throw(PyExc_ValueError, "the state tuple must have a single element");
        }
        auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
        if (!ptr) {
            pygmo_throw(PyExc_TypeError, "a bytes object is needed to deserialize a problem");
        }
        const auto size = len(state[0]);
        std::string s(ptr, ptr + size);
        std::istringstream iss;
        iss.str(s);
        {
            cereal::PortableBinaryInputArchive iarchive(iss);
            iarchive(a);
        }
    }
};
}

#endif
