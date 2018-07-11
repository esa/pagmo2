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

#ifndef PYGMO_ALGORITHM_HPP
#define PYGMO_ALGORITHM_HPP

#include <pygmo/python_includes.hpp>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/tuple.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/population.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>

#include <pygmo/common_base.hpp>
#include <pygmo/common_utils.hpp>
#include <pygmo/object_serialization.hpp>

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

// Disable the static UDA checks for bp::object.
template <>
struct disable_uda_checks<bp::object> : std::true_type {
};

template <>
struct algo_inner<bp::object> final : algo_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    algo_inner() = default;
    algo_inner(const algo_inner &) = delete;
    algo_inner(algo_inner &&) = delete;
    algo_inner &operator=(const algo_inner &) = delete;
    algo_inner &operator=(algo_inner &&) = delete;
    explicit algo_inner(const bp::object &o)
    {
        check_not_type(o, "algorithm");
        check_mandatory_method(o, "evolve", "algorithm");
        m_value = pygmo::deepcopy(o);
    }
    virtual std::unique_ptr<algo_inner_base> clone() const override final
    {
        // This will make a deep copy using the ctor above.
        return make_unique<algo_inner>(m_value);
    }
    // Mandatory methods.
    virtual population evolve(const population &pop) const override final
    {
        return bp::extract<population>(m_value.attr("evolve")(pop));
    }
    // Optional methods.
    virtual void set_seed(unsigned n) override final
    {
        auto ss = pygmo::callable_attribute(m_value, "set_seed");
        if (ss.is_none()) {
            pygmo_throw(PyExc_NotImplementedError,
                        ("set_seed() has been invoked but it is not implemented "
                         "in the user-defined Python algorithm '"
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the method is either not present or not callable")
                            .c_str());
        }
        ss(n);
    }
    virtual bool has_set_seed() const override final
    {
        auto ss = pygmo::callable_attribute(m_value, "set_seed");
        if (ss.is_none()) {
            return false;
        }
        auto hss = pygmo::callable_attribute(m_value, "has_set_seed");
        if (hss.is_none()) {
            return true;
        }
        return bp::extract<bool>(hss());
    }
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
    virtual void set_verbosity(unsigned n) override final
    {
        auto sv = pygmo::callable_attribute(m_value, "set_verbosity");
        if (sv.is_none()) {
            pygmo_throw(PyExc_NotImplementedError,
                        ("set_verbosity() has been invoked but it is not implemented "
                         "in the user-defined Python algorithm '"
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the method is either not present or not callable")
                            .c_str());
        }
        sv(n);
    }
    virtual bool has_set_verbosity() const override final
    {
        auto sv = pygmo::callable_attribute(m_value, "set_verbosity");
        if (sv.is_none()) {
            return false;
        }
        auto hsv = pygmo::callable_attribute(m_value, "has_set_verbosity");
        if (hsv.is_none()) {
            return true;
        }
        return bp::extract<bool>(hsv());
    }
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<algo_inner_base>(this), m_value);
    }
    bp::object m_value;
};
} // namespace detail
} // namespace pagmo

// Register the algo_inner specialisation for bp::object.
PAGMO_REGISTER_ALGORITHM(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

// Serialization support for the algorithm class.
struct algorithm_pickle_suite : bp::pickle_suite {
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
        // Store the cerealized algorithm plus the list of currently-loaded APs.
        return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())), get_ap_list());
    }
    static void setstate(pagmo::algorithm &a, const bp::tuple &state)
    {
        // Similarly, first we extract a bytes object from the Python state,
        // and then we build a C++ string from it. The string is then used
        // to decerealise the object.
        if (len(state) != 2) {
            pygmo_throw(PyExc_ValueError, ("the state tuple passed for algorithm deserialization "
                                           "must have 2 elements, but instead it has "
                                           + std::to_string(len(state)) + " elements")
                                              .c_str());
        }

        // Make sure we import all the aps specified in the archive.
        import_aps(bp::list(state[1]));

        auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
        if (!ptr) {
            pygmo_throw(PyExc_TypeError, "a bytes object is needed to deserialize an algorithm");
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
} // namespace pygmo

#endif
