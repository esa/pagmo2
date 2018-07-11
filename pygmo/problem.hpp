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

#ifndef PYGMO_PROBLEM_HPP
#define PYGMO_PROBLEM_HPP

#include <pygmo/python_includes.hpp>

#include <algorithm>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/class.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/tuple.hpp>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

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

// Disable the static UDP checks for bp::object.
template <>
struct disable_udp_checks<bp::object> : std::true_type {
};

// NOTE: here we are specialising the prob_inner implementation template for bp::object.
// We need to do this because the default implementation works on C++ types by detecting
// their methods via type-traits at compile-time, but here we need to check the presence
// of methods at runtime. That is, we need to replace the type-traits with runtime
// inspection of Python objects.
//
// We cannot be as precise as in C++ detecting the methods' signatures (it might be
// possible with the inspect module in principle, but it looks messy and it might break if the methods
// are implemented as C/C++ extensions). The main policy adopted here is: if the bp::object
// has a callable attribute with the required name, then the "runtime type-trait" is considered
// satisfied, otherwise not.
template <>
struct prob_inner<bp::object> final : prob_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    prob_inner() = default;
    prob_inner(const prob_inner &) = delete;
    prob_inner(prob_inner &&) = delete;
    prob_inner &operator=(const prob_inner &) = delete;
    prob_inner &operator=(prob_inner &&) = delete;
    explicit prob_inner(const bp::object &o)
    {
        check_not_type(o, "problem");
        // Check the presence of the mandatory methods (these are static asserts
        // in the C++ counterpart).
        check_mandatory_method(o, "fitness", "problem");
        check_mandatory_method(o, "get_bounds", "problem");
        // The Python UDP looks alright, let's deepcopy it into m_value.
        m_value = pygmo::deepcopy(o);
    }
    virtual std::unique_ptr<prob_inner_base> clone() const override final
    {
        // This will make a deep copy using the ctor above.
        return make_unique<prob_inner>(m_value);
    }
    // Mandatory methods.
    virtual vector_double fitness(const vector_double &dv) const override final
    {
        return pygmo::to_vd(m_value.attr("fitness")(pygmo::v_to_a(dv)));
    }
    virtual std::pair<vector_double, vector_double> get_bounds() const override final
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
    // Optional methods.
    virtual vector_double::size_type get_nobj() const override final
    {
        return getter_wrapper<vector_double::size_type>(m_value, "get_nobj", 1u);
    }
    virtual vector_double::size_type get_nec() const override final
    {
        return getter_wrapper<vector_double::size_type>(m_value, "get_nec", 0u);
    }
    virtual vector_double::size_type get_nic() const override final
    {
        return getter_wrapper<vector_double::size_type>(m_value, "get_nic", 0u);
    }
    virtual vector_double::size_type get_nix() const override final
    {
        return getter_wrapper<vector_double::size_type>(m_value, "get_nix", 0u);
    }
    virtual std::string get_name() const override final
    {
        return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
    }
    virtual std::string get_extra_info() const override final
    {
        return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
    }
    virtual bool has_gradient() const override final
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
    virtual vector_double gradient(const vector_double &dv) const override final
    {
        auto g = pygmo::callable_attribute(m_value, "gradient");
        if (g.is_none()) {
            pygmo_throw(PyExc_NotImplementedError,
                        ("gradients have been requested but they are not implemented "
                         "in the user-defined Python problem '"
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the method is either not present or not callable")
                            .c_str());
        }
        return pygmo::to_vd(g(pygmo::v_to_a(dv)));
    }
    virtual bool has_gradient_sparsity() const override final
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
    virtual sparsity_pattern gradient_sparsity() const override final
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
    virtual bool has_hessians() const override final
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
    virtual std::vector<vector_double> hessians(const vector_double &dv) const override final
    {
        auto h = pygmo::callable_attribute(m_value, "hessians");
        if (h.is_none()) {
            pygmo_throw(PyExc_NotImplementedError,
                        ("hessians have been requested but they are not implemented "
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
    virtual bool has_hessians_sparsity() const override final
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
    virtual std::vector<sparsity_pattern> hessians_sparsity() const override final
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
    virtual void set_seed(unsigned n) override final
    {
        auto ss = pygmo::callable_attribute(m_value, "set_seed");
        if (ss.is_none()) {
            pygmo_throw(PyExc_NotImplementedError,
                        ("set_seed() has been invoked but it is not implemented "
                         "in the user-defined Python problem '"
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the method is either not present or not callable")
                            .c_str());
        }
        ss(n);
    }
    virtual bool has_set_seed() const override final
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
    virtual pagmo::thread_safety get_thread_safety() const override final
    {
        return pagmo::thread_safety::none;
    }
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<prob_inner_base>(this), m_value);
    }
    bp::object m_value;
};
} // namespace detail
} // namespace pagmo

// Register the prob_inner specialisation for bp::object.
PAGMO_REGISTER_PROBLEM(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

// Serialization support for the problem class.
struct problem_pickle_suite : bp::pickle_suite {
    static bp::tuple getstate(const pagmo::problem &p)
    {
        // The idea here is that first we extract a char array
        // into which problem has been cerealised, then we turn
        // this object into a Python bytes object and return that.
        std::ostringstream oss;
        {
            cereal::PortableBinaryOutputArchive oarchive(oss);
            oarchive(p);
        }
        auto s = oss.str();
        return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())), get_ap_list());
    }
    static void setstate(pagmo::problem &p, const bp::tuple &state)
    {
        // Similarly, first we extract a bytes object from the Python state,
        // and then we build a C++ string from it. The string is then used
        // to decerealise the object.
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
            cereal::PortableBinaryInputArchive iarchive(iss);
            iarchive(p);
        }
    }
};
} // namespace pygmo

#endif
