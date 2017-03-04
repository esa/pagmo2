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
#include <boost/python/extract.hpp>
#include <boost/python/object.hpp>
#include <boost/python/tuple.hpp>
#include <sstream>
#include <string>
#include <type_traits>

#include <pagmo/algorithm.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/island.hpp>
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

// Disable the static UDI checks for bp::object.
template <>
struct disable_udi_checks<bp::object> : std::true_type {
};

template <>
struct isl_inner<bp::object> final : isl_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    isl_inner() = default;
    isl_inner(const isl_inner &) = delete;
    isl_inner(isl_inner &&) = delete;
    isl_inner &operator=(const isl_inner &) = delete;
    isl_inner &operator=(isl_inner &&) = delete;
    explicit isl_inner(const bp::object &o)
    {
        check_mandatory_method(o, "enqueue_evolution", "island");
        check_mandatory_method(o, "wait", "island");
        check_mandatory_method(o, "get_population", "island");
        m_value = pygmo::deepcopy(o);
    }
    virtual isl_inner_base *clone() const override final
    {
        // This will make a deep copy using the ctor above.
        return ::new isl_inner(m_value);
    }
    // Mandatory methods.
    virtual void enqueue_evolution(const algorithm &algo, archipelago *archi) override final
    {
        // NOTE: here Boost Python will create a copy of algo wrapped in a bp::object before
        // passing it to the Python method.
        if (archi) {
            m_value.attr("enqueue_evolution")(algo, *archi);
        } else {
            m_value.attr("enqueue_evolution")(algo, bp::object());
        }
    }
    virtual void wait() const override final
    {
        m_value.attr("wait")();
    }
    virtual population get_population() const override final
    {
        return bp::extract<population>(m_value.attr("get_population")());
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
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<isl_inner_base>(this), m_value);
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
        // into which isl has been cerealised, then we turn
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
