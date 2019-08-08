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

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/import.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/tuple.hpp>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/handle_thread_py_exception.hpp>
#include <pygmo/topology.hpp>

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

topo_inner<bp::object>::topo_inner(const bp::object &o)
{
    // Forbid the use of a pygmo.topology as a UDT.
    // The motivation here is consistency with C++. In C++, the use of
    // a pagmo::topology as a UDT is forbidden and prevented by the fact
    // that the generic constructor from UDT is disabled if the input
    // object is a pagmo::topology (the copy/move constructor is
    // invoked instead). In order to achieve an equivalent behaviour
    // in pygmo, we throw an error if o is a topology, and instruct
    // the user to employ the standard copy/deepcopy facilities
    // for creating a copy of the input topology.
    if (pygmo::type(o) == bp::import("pygmo").attr("topology")) {
        pygmo_throw(PyExc_TypeError,
                    ("a pygmo.topology cannot be used as a UDT for another pygmo.topology (if you need to copy a "
                     "topology please use the standard Python copy()/deepcopy() functions)"));
    }
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "topology");
    check_mandatory_method(o, "get_connections", "topology");
    check_mandatory_method(o, "push_back", "topology");
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<topo_inner_base> topo_inner<bp::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return detail::make_unique<topo_inner>(m_value);
}

std::pair<std::vector<std::size_t>, vector_double> topo_inner<bp::object>::get_connections(std::size_t n) const
{
    // NOTE: get_connections() may be called from a separate thread in pagmo::island, need to construct a GTE before
    // doing anything with the interpreter (including the throws in the checks below).
    pygmo::gil_thread_ensurer gte;

    // NOTE: every time we call into the Python interpreter from a separate thread, we need to
    // handle Python exceptions in a special way.
    std::string topo_name;
    try {
        topo_name = get_name();
    } catch (const bp::error_already_set &) {
        pygmo::handle_thread_py_exception("Could not fetch the name of a pythonic topology. The error is:\n");
    }

    try {
        // Fetch the connections in Python form.
        bp::object o = m_value.attr("get_connections")(n);

        // Prepare the return value.
        std::pair<std::vector<std::size_t>, vector_double> retval;

        // We will try to interpret o as a collection of generic python objects.
        bp::stl_input_iterator<bp::object> begin(o), end;

        if (begin == end) {
            // Empty iteratable.
            pygmo_throw(PyExc_ValueError, ("the iteratable returned by a topology of type '" + topo_name
                                           + "' is empty (it should contain 2 elements)")
                                              .c_str());
        }

        retval.first = pygmo::to_vuint<std::size_t>(*begin);

        if (++begin == end) {
            // Only one element in the iteratable.
            pygmo_throw(PyExc_ValueError, ("the iteratable returned by a topology of type '" + topo_name
                                           + "' has only 1 element (it should contain 2 elements)")
                                              .c_str());
        }

        retval.second = pygmo::to_vd(*begin);

        if (++begin != end) {
            // Too many elements.
            pygmo_throw(PyExc_ValueError, ("the iteratable returned by a topology of type '" + topo_name
                                           + "' has more than 2 elements (it should contain 2 elements)")
                                              .c_str());
        }

        return retval;
    } catch (const bp::error_already_set &) {
        pygmo::handle_thread_py_exception("The get_connections() method of a pythonic topology of type '" + topo_name
                                          + "' raised an error:\n");
    }
}

void topo_inner<bp::object>::push_back()
{
    m_value.attr("push_back")();
}

std::string topo_inner<bp::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string topo_inner<bp::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_TOPOLOGY_IMPLEMENT(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

bp::tuple topology_pickle_suite::getstate(const pagmo::topology &t)
{
    // The idea here is that first we extract a char array
    // into which t has been serialized, then we turn
    // this object into a Python bytes object and return that.
    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oarchive(oss);
        oarchive << t;
    }
    auto s = oss.str();
    // Store the serialized topology plus the list of currently-loaded APs.
    return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())), get_ap_list());
}

void topology_pickle_suite::setstate(pagmo::topology &t, const bp::tuple &state)
{
    // Similarly, first we extract a bytes object from the Python state,
    // and then we build a C++ string from it. The string is then used
    // to deserialize the object.
    if (len(state) != 2) {
        pygmo_throw(PyExc_ValueError, ("the state tuple passed for topology deserialization "
                                       "must have 2 elements, but instead it has "
                                       + std::to_string(len(state)) + " elements")
                                          .c_str());
    }

    // Make sure we import all the aps specified in the archive.
    import_aps(bp::list(state[1]));

    auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
    if (!ptr) {
        pygmo_throw(PyExc_TypeError, "a bytes object is needed to deserialize a topology");
    }
    const auto size = len(state[0]);
    std::string s(ptr, ptr + size);
    std::istringstream iss;
    iss.str(s);
    {
        boost::archive::binary_iarchive iarchive(iss);
        iarchive >> t;
    }
}

} // namespace pygmo
