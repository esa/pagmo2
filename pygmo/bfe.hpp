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

#ifndef PYGMO_BFE_HPP
#define PYGMO_BFE_HPP

#include <pygmo/python_includes.hpp>

#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/tuple.hpp>

#include <pagmo/bfe.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#include <pygmo/common_base.hpp>
#include <pygmo/common_utils.hpp>
#include <pygmo/object_serialization.hpp>

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

// Disable the static UDBFE checks for bp::object.
template <>
struct disable_udbfe_checks<bp::object> : std::true_type {
};

template <>
struct bfe_inner<bp::object> final : bfe_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    bfe_inner() = default;
    bfe_inner(const bfe_inner &) = delete;
    bfe_inner(bfe_inner &&) = delete;
    bfe_inner &operator=(const bfe_inner &) = delete;
    bfe_inner &operator=(bfe_inner &&) = delete;
    explicit bfe_inner(const bp::object &o)
    {
        // Forbid the use of a pygmo.bfe as a UDBFE.
        // The motivation here is consistency with C++. In C++, the use of
        // a pagmo::bfe as a UDBFE is forbidden and prevented by the fact
        // that the generic constructor from UDBFE is disabled if the input
        // object is a pagmo::bfe (the copy/move constructor is
        // invoked instead). In order to achieve an equivalent behaviour
        // in pygmo, we throw an error if o is a bfe, and instruct
        // the user to employ the standard copy/deepcopy facilities
        // for creating a copy of the input bfe.
        if (pygmo::type(o) == bp::import("pygmo").attr("bfe")) {
            pygmo_throw(PyExc_TypeError,
                        ("a pygmo.bfe cannot be used as a UDBFE for another pygmo.bfe (if you need to copy a "
                         "bfe please use the standard Python copy()/deepcopy() functions)"));
        }
        // Check that o is an instance of a class, and not a type.
        check_not_type(o, "bfe");
        check_mandatory_method(o, "__call__", "bfe");
        m_value = pygmo::deepcopy(o);
    }
    virtual std::unique_ptr<bfe_inner_base> clone() const override final
    {
        // This will make a deep copy using the ctor above.
        return detail::make_unique<bfe_inner>(m_value);
    }
    // Mandatory methods.
    virtual vector_double operator()(const problem &p, const vector_double &dvs) const override final
    {
        return pygmo::to_vd(m_value.attr("__call__")(p, pygmo::v_to_a(dvs)));
    }
    // Optional methods.
    virtual pagmo::thread_safety get_thread_safety() const override final
    {
        return pagmo::thread_safety::none;
    }
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
        ar(cereal::base_class<bfe_inner_base>(this), m_value);
    }
    bp::object m_value;
};
} // namespace detail
} // namespace pagmo

// Register the bfe_inner specialisation for bp::object.
PAGMO_REGISTER_BFE(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

// Serialization support for the bfe class.
struct bfe_pickle_suite : bp::pickle_suite {
    static bp::tuple getstate(const pagmo::bfe &b)
    {
        // The idea here is that first we extract a char array
        // into which bfe has been cerealised, then we turn
        // this object into a Python bytes object and return that.
        std::ostringstream oss;
        {
            cereal::PortableBinaryOutputArchive oarchive(oss);
            oarchive(b);
        }
        auto s = oss.str();
        // Store the cerealized bfe plus the list of currently-loaded APs.
        return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())), get_ap_list());
    }
    static void setstate(pagmo::bfe &b, const bp::tuple &state)
    {
        // Similarly, first we extract a bytes object from the Python state,
        // and then we build a C++ string from it. The string is then used
        // to decerealise the object.
        if (len(state) != 2) {
            pygmo_throw(PyExc_ValueError, ("the state tuple passed for bfe deserialization "
                                           "must have 2 elements, but instead it has "
                                           + std::to_string(len(state)) + " elements")
                                              .c_str());
        }

        // Make sure we import all the aps specified in the archive.
        import_aps(bp::list(state[1]));

        auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
        if (!ptr) {
            pygmo_throw(PyExc_TypeError, "a bytes object is needed to deserialize a bfe");
        }
        const auto size = len(state[0]);
        std::string s(ptr, ptr + size);
        std::istringstream iss;
        iss.str(s);
        {
            cereal::PortableBinaryInputArchive iarchive(iss);
            iarchive(b);
        }
    }
};
} // namespace pygmo

#endif
