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

#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/import.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/tuple.hpp>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/problem.hpp>

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

prob_inner<bp::object>::prob_inner(const bp::object &o)
{
    // Forbid the use of a pygmo.problem as a UDP.
    // The motivation here is consistency with C++. In C++, the use of
    // a pagmo::problem as a UDP is forbidden and prevented by the fact
    // that the generic constructor from UDP is disabled if the input
    // object is a pagmo::problem (the copy/move constructor is
    // invoked instead). In order to achieve an equivalent behaviour
    // in pygmo, we throw an error if o is a problem, and instruct
    // the user to employ the standard copy/deepcopy facilities
    // for creating a copy of the input problem.
    if (pygmo::type(o) == bp::import("pygmo").attr("problem")) {
        pygmo_throw(PyExc_TypeError,
                    ("a pygmo.problem cannot be used as a UDP for another pygmo.problem (if you need to copy a "
                     "problem please use the standard Python copy()/deepcopy() functions)"));
    }
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "problem");
    // Check the presence of the mandatory methods (these are static asserts
    // in the C++ counterpart).
    check_mandatory_method(o, "fitness", "problem");
    check_mandatory_method(o, "get_bounds", "problem");
    // The Python UDP looks alright, let's deepcopy it into m_value.
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<prob_inner_base> prob_inner<bp::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return detail::make_unique<prob_inner>(m_value);
}

vector_double prob_inner<bp::object>::fitness(const vector_double &dv) const
{
    return pygmo::to_vd(m_value.attr("fitness")(pygmo::v_to_a(dv)));
}

std::pair<vector_double, vector_double> prob_inner<bp::object>::get_bounds() const
{
    bp::tuple tup = bp::extract<bp::tuple>(m_value.attr("get_bounds")());
    // Check the tuple size.
    if (len(tup) != 2) {
        pygmo_throw(PyExc_ValueError, ("the bounds of the problem must be returned as a tuple of 2 elements, but "
                                       "the detected tuple size is "
                                       + std::to_string(len(tup)))
                                          .c_str());
    }
    // Finally, we build the pair from the tuple elements.
    return std::make_pair(pygmo::to_vd(tup[0]), pygmo::to_vd(tup[1]));
}

vector_double prob_inner<bp::object>::batch_fitness(const vector_double &dv) const
{
    auto bf = pygmo::callable_attribute(m_value, "batch_fitness");
    if (bf.is_none()) {
        pygmo_throw(PyExc_NotImplementedError,
                    ("the batch_fitness() method has been invoked, but it is not implemented "
                     "in the user-defined Python problem '"
                     + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                     + "': the method is either not present or not callable")
                        .c_str());
    }
    return pygmo::to_vd(bf(pygmo::v_to_a(dv)));
}

bool prob_inner<bp::object>::has_batch_fitness() const
{
    // Same logic as in C++:
    // - without a batch_fitness() method, return false;
    // - with a batch_fitness() and no override, return true;
    // - with a batch_fitness() and override, return the value from the override.
    auto bf = pygmo::callable_attribute(m_value, "batch_fitness");
    if (bf.is_none()) {
        return false;
    }
    auto hbf = pygmo::callable_attribute(m_value, "has_batch_fitness");
    if (hbf.is_none()) {
        return true;
    }
    return bp::extract<bool>(hbf());
}

vector_double::size_type prob_inner<bp::object>::get_nobj() const
{
    return getter_wrapper<vector_double::size_type>(m_value, "get_nobj", 1u);
}

vector_double::size_type prob_inner<bp::object>::get_nec() const
{
    return getter_wrapper<vector_double::size_type>(m_value, "get_nec", 0u);
}

vector_double::size_type prob_inner<bp::object>::get_nic() const
{
    return getter_wrapper<vector_double::size_type>(m_value, "get_nic", 0u);
}

vector_double::size_type prob_inner<bp::object>::get_nix() const
{
    return getter_wrapper<vector_double::size_type>(m_value, "get_nix", 0u);
}

std::string prob_inner<bp::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string prob_inner<bp::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

bool prob_inner<bp::object>::has_gradient() const
{
    // Same logic as in C++:
    // - without a gradient() method, return false;
    // - with a gradient() and no override, return true;
    // - with a gradient() and override, return the value from the override.
    auto g = pygmo::callable_attribute(m_value, "gradient");
    if (g.is_none()) {
        return false;
    }
    auto hg = pygmo::callable_attribute(m_value, "has_gradient");
    if (hg.is_none()) {
        return true;
    }
    return bp::extract<bool>(hg());
}

vector_double prob_inner<bp::object>::gradient(const vector_double &dv) const
{
    auto g = pygmo::callable_attribute(m_value, "gradient");
    if (g.is_none()) {
        pygmo_throw(PyExc_NotImplementedError, ("gradients have been requested but they are not implemented "
                                                "in the user-defined Python problem '"
                                                + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                                                + "': the method is either not present or not callable")
                                                   .c_str());
    }
    return pygmo::to_vd(g(pygmo::v_to_a(dv)));
}

bool prob_inner<bp::object>::has_gradient_sparsity() const
{
    // Same logic as in C++:
    // - without a gradient_sparsity() method, return false;
    // - with a gradient_sparsity() and no override, return true;
    // - with a gradient_sparsity() and override, return the value from the override.
    auto gs = pygmo::callable_attribute(m_value, "gradient_sparsity");
    if (gs.is_none()) {
        return false;
    }
    auto hgs = pygmo::callable_attribute(m_value, "has_gradient_sparsity");
    if (hgs.is_none()) {
        return true;
    }
    return bp::extract<bool>(hgs());
}

sparsity_pattern prob_inner<bp::object>::gradient_sparsity() const
{
    auto gs = pygmo::callable_attribute(m_value, "gradient_sparsity");
    if (gs.is_none()) {
        // NOTE: this is similar to C++: this virtual method gradient_sparsity() we are in, is called
        // only if the availability of gradient_sparsity() in the UDP was detected upon the construction
        // of a problem (i.e., m_has_gradient_sparsity is set to true). If the UDP didn't have a gradient_sparsity()
        // method upon problem construction, the m_has_gradient_sparsity is set to false and we never get here.
        // However, in Python we could have a situation in which a method is erased at runtime, so it is
        // still possible to end up in this point (if gradient_sparsity() in the internal UDP was erased
        // after the problem construction). This is something we need to strongly discourage, hence the message.
        pygmo_throw(PyExc_RuntimeError,
                    ("gradient sparsity has been requested but it is not implemented."
                     "This indicates a logical error in the implementation of the user-defined Python problem "
                     + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                     + "': the gradient sparsity was available at problem construction but it has been removed "
                       "at a later stage")
                        .c_str());
    }
    return pygmo::to_sp(gs());
}

bool prob_inner<bp::object>::has_hessians() const
{
    // Same logic as in C++:
    // - without a hessians() method, return false;
    // - with a hessians() and no override, return true;
    // - with a hessians() and override, return the value from the override.
    auto h = pygmo::callable_attribute(m_value, "hessians");
    if (h.is_none()) {
        return false;
    }
    auto hh = pygmo::callable_attribute(m_value, "has_hessians");
    if (hh.is_none()) {
        return true;
    }
    return bp::extract<bool>(hh());
}

std::vector<vector_double> prob_inner<bp::object>::hessians(const vector_double &dv) const
{
    auto h = pygmo::callable_attribute(m_value, "hessians");
    if (h.is_none()) {
        pygmo_throw(PyExc_NotImplementedError, ("hessians have been requested but they are not implemented "
                                                "in the user-defined Python problem '"
                                                + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                                                + "': the method is either not present or not callable")
                                                   .c_str());
    }
    // Invoke the method, getting out a generic Python object.
    bp::object tmp = h(pygmo::v_to_a(dv));
    // Let's build the return value.
    std::vector<vector_double> retval;
    bp::stl_input_iterator<bp::object> begin(tmp), end;
    std::transform(begin, end, std::back_inserter(retval), [](const bp::object &o) { return pygmo::to_vd(o); });
    return retval;
}

bool prob_inner<bp::object>::has_hessians_sparsity() const
{
    // Same logic as in C++:
    // - without a hessians_sparsity() method, return false;
    // - with a hessians_sparsity() and no override, return true;
    // - with a hessians_sparsity() and override, return the value from the override.
    auto hs = pygmo::callable_attribute(m_value, "hessians_sparsity");
    if (hs.is_none()) {
        return false;
    }
    auto hhs = pygmo::callable_attribute(m_value, "has_hessians_sparsity");
    if (hhs.is_none()) {
        return true;
    }
    return bp::extract<bool>(hhs());
}

std::vector<sparsity_pattern> prob_inner<bp::object>::hessians_sparsity() const
{
    auto hs = pygmo::callable_attribute(m_value, "hessians_sparsity");
    if (hs.is_none()) {
        pygmo_throw(PyExc_RuntimeError,
                    ("hessians sparsity has been requested but it is not implemented."
                     "This indicates a logical error in the implementation of the user-defined Python problem "
                     + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                     + "': the hessians sparsity was available at problem construction but it has been removed "
                       "at a later stage")
                        .c_str());
    }
    bp::object tmp = hs();
    std::vector<sparsity_pattern> retval;
    bp::stl_input_iterator<bp::object> begin(tmp), end;
    std::transform(begin, end, std::back_inserter(retval), [](const bp::object &o) { return pygmo::to_sp(o); });
    return retval;
}

void prob_inner<bp::object>::set_seed(unsigned n)
{
    auto ss = pygmo::callable_attribute(m_value, "set_seed");
    if (ss.is_none()) {
        pygmo_throw(PyExc_NotImplementedError, ("set_seed() has been invoked but it is not implemented "
                                                "in the user-defined Python problem '"
                                                + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                                                + "': the method is either not present or not callable")
                                                   .c_str());
    }
    ss(n);
}

bool prob_inner<bp::object>::has_set_seed() const
{
    // Same logic as in C++:
    // - without a set_seed() method, return false;
    // - with a set_seed() and no override, return true;
    // - with a set_seed() and override, return the value from the override.
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

// Hard code no thread safety for python problems.
pagmo::thread_safety prob_inner<bp::object>::get_thread_safety() const
{
    return pagmo::thread_safety::none;
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

// Serialization support for the problem class.
bp::tuple problem_pickle_suite::getstate(const pagmo::problem &p)
{
    // The idea here is that first we extract a char array
    // into which problem has been serialized, then we turn
    // this object into a Python bytes object and return that.
    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oarchive(oss);
        oarchive << p;
    }
    auto s = oss.str();
    return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())), get_ap_list());
}

void problem_pickle_suite::setstate(pagmo::problem &p, const bp::tuple &state)
{
    // Similarly, first we extract a bytes object from the Python state,
    // and then we build a C++ string from it. The string is then used
    // to deserialized the object.
    if (len(state) != 2) {
        pygmo_throw(PyExc_ValueError, ("the state tuple passed for problem deserialization "
                                       "must have 2 elements, but instead it has "
                                       + std::to_string(len(state)) + " elements")
                                          .c_str());
    }

    // Make sure we import all the aps specified in the archive.
    import_aps(bp::list(state[1]));

    auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
    if (!ptr) {
        pygmo_throw(PyExc_TypeError, "a bytes object is needed to deserialize a problem");
    }
    const auto size = len(state[0]);
    std::string s(ptr, ptr + size);
    std::istringstream iss;
    iss.str(s);
    {
        boost::archive::binary_iarchive iarchive(iss);
        iarchive >> p;
    }
}

} // namespace pygmo
