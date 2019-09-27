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
#include <boost/python/errors.hpp>
#include <boost/python/import.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/tuple.hpp>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/s_policy.hpp>
#include <pagmo/types.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/handle_thread_py_exception.hpp>
#include <pygmo/s_policy.hpp>

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

s_pol_inner<bp::object>::s_pol_inner(const bp::object &o)
{
    // Forbid the use of a pygmo.s_policy as a UDSP.
    // The motivation here is consistency with C++. In C++, the use of
    // a pagmo::s_policy as a UDSP is forbidden and prevented by the fact
    // that the generic constructor from UDSP is disabled if the input
    // object is a pagmo::s_policy (the copy/move constructor is
    // invoked instead). In order to achieve an equivalent behaviour
    // in pygmo, we throw an error if o is an s_policy, and instruct
    // the user to employ the standard copy/deepcopy facilities
    // for creating a copy of the input s_policy.
    if (pygmo::type(o) == bp::import("pygmo").attr("s_policy")) {
        pygmo_throw(PyExc_TypeError,
                    ("a pygmo.s_policy cannot be used as a UDSP for another pygmo.s_policy (if you need to copy a "
                     "selection policy please use the standard Python copy()/deepcopy() functions)"));
    }
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "s_policy");
    check_mandatory_method(o, "select", "s_policy");
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<s_pol_inner_base> s_pol_inner<bp::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return detail::make_unique<s_pol_inner>(m_value);
}

individuals_group_t s_pol_inner<bp::object>::select(const individuals_group_t &inds, const vector_double::size_type &nx,
                                                    const vector_double::size_type &nix,
                                                    const vector_double::size_type &nobj,
                                                    const vector_double::size_type &nec,
                                                    const vector_double::size_type &nic, const vector_double &tol) const
{
    // NOTE: select() may be called from a separate thread in pagmo::island, need to construct a GTE before
    // doing anything with the interpreter (including the throws in the checks below).
    pygmo::gil_thread_ensurer gte;

    // NOTE: every time we call into the Python interpreter from a separate thread, we need to
    // handle Python exceptions in a special way.
    std::string s_pol_name;
    try {
        s_pol_name = get_name();
    } catch (const bp::error_already_set &) {
        pygmo::handle_thread_py_exception("Could not fetch the name of a pythonic selection policy. The error is:\n");
    }

    try {
        // Fetch the new individuals in Python form.
        bp::object o
            = m_value.attr("select")(pygmo::inds_to_tuple(inds), nx, nix, nobj, nec, nic, pygmo::vector_to_ndarr(tol));

        // Convert back to C++ form and return.
        return pygmo::obj_to_inds(o);
    } catch (const bp::error_already_set &) {
        pygmo::handle_thread_py_exception("The select() method of a pythonic selection policy of type '" + s_pol_name
                                          + "' raised an error:\n");
    }
}

std::string s_pol_inner<bp::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string s_pol_inner<bp::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_S_POLICY_IMPLEMENT(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

bp::tuple s_policy_pickle_suite::getstate(const pagmo::s_policy &sp)
{
    // The idea here is that first we extract a char array
    // into which t has been serialized, then we turn
    // this object into a Python bytes object and return that.
    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oarchive(oss);
        oarchive << sp;
    }
    auto s = oss.str();
    // Store the serialized s_policy plus the list of currently-loaded APs.
    return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())), get_ap_list());
}

void s_policy_pickle_suite::setstate(pagmo::s_policy &sp, const bp::tuple &state)
{
    // Similarly, first we extract a bytes object from the Python state,
    // and then we build a C++ string from it. The string is then used
    // to deserialize the object.
    if (len(state) != 2) {
        pygmo_throw(PyExc_ValueError, ("the state tuple passed for s_policy deserialization "
                                       "must have 2 elements, but instead it has "
                                       + std::to_string(len(state)) + " elements")
                                          .c_str());
    }

    // Make sure we import all the aps specified in the archive.
    import_aps(bp::list(state[1]));

    auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
    if (!ptr) {
        pygmo_throw(PyExc_TypeError, "a bytes object is needed to deserialize a s_policy");
    }
    const auto size = len(state[0]);
    std::string s(ptr, ptr + size);
    std::istringstream iss;
    iss.str(s);
    {
        boost::archive::binary_iarchive iarchive(iss);
        iarchive >> sp;
    }
}

} // namespace pygmo
