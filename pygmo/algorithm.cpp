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

#include <memory>
#include <sstream>
#include <string>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/import.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/tuple.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>

#include <pygmo/algorithm.hpp>
#include <pygmo/common_utils.hpp>

namespace bp = boost::python;

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

algo_inner<bp::object>::algo_inner(const bp::object &o)
{
    // Forbid the use of a pygmo.algorithm as a UDA.
    // The motivation here is consistency with C++. In C++, the use of
    // a pagmo::algorithm as a UDA is forbidden and prevented by the fact
    // that the generic constructor from UDA is disabled if the input
    // object is a pagmo::algorithm (the copy/move constructor is
    // invoked instead). In order to achieve an equivalent behaviour
    // in pygmo, we throw an error if o is a algorithm, and instruct
    // the user to employ the standard copy/deepcopy facilities
    // for creating a copy of the input algorithm.
    if (pygmo::type(o) == bp::import("pygmo").attr("algorithm")) {
        pygmo_throw(PyExc_TypeError,
                    ("a pygmo.algorithm cannot be used as a UDA for another pygmo.algorithm (if you need to copy an "
                     "algorithm please use the standard Python copy()/deepcopy() functions)"));
    }
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "algorithm");
    check_mandatory_method(o, "evolve", "algorithm");
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<algo_inner_base> algo_inner<bp::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return detail::make_unique<algo_inner>(m_value);
}

population algo_inner<bp::object>::evolve(const population &pop) const
{
    return bp::extract<population>(m_value.attr("evolve")(pop));
}

void algo_inner<bp::object>::set_seed(unsigned n)
{
    auto ss = pygmo::callable_attribute(m_value, "set_seed");
    if (ss.is_none()) {
        pygmo_throw(PyExc_NotImplementedError, ("set_seed() has been invoked but it is not implemented "
                                                "in the user-defined Python algorithm '"
                                                + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                                                + "': the method is either not present or not callable")
                                                   .c_str());
    }
    ss(n);
}

bool algo_inner<bp::object>::has_set_seed() const
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

pagmo::thread_safety algo_inner<bp::object>::get_thread_safety() const
{
    return pagmo::thread_safety::none;
}

std::string algo_inner<bp::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string algo_inner<bp::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

void algo_inner<bp::object>::set_verbosity(unsigned n)
{
    auto sv = pygmo::callable_attribute(m_value, "set_verbosity");
    if (sv.is_none()) {
        pygmo_throw(PyExc_NotImplementedError, ("set_verbosity() has been invoked but it is not implemented "
                                                "in the user-defined Python algorithm '"
                                                + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                                                + "': the method is either not present or not callable")
                                                   .c_str());
    }
    sv(n);
}

bool algo_inner<bp::object>::has_set_verbosity() const
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

} // namespace detail

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

bp::tuple algorithm_pickle_suite::getstate(const pagmo::algorithm &a)
{
    // The idea here is that first we extract a char array
    // into which algorithm has been serialized, then we turn
    // this object into a Python bytes object and return that.
    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oarchive(oss);
        oarchive << a;
    }
    auto s = oss.str();
    // Store the serialized algorithm plus the list of currently-loaded APs.
    return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())), get_ap_list());
}

void algorithm_pickle_suite::setstate(pagmo::algorithm &a, const bp::tuple &state)
{
    // Similarly, first we extract a bytes object from the Python state,
    // and then we build a C++ string from it. The string is then used
    // to deserialize the object.
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
        boost::archive::binary_iarchive iarchive(iss);
        iarchive >> a;
    }
}

} // namespace pygmo
