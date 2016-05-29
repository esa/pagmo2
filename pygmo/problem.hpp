#ifndef PYGMO_PROBLEM_HPP
#define PYGMO_PROBLEM_HPP

#include "python_includes.hpp"

#include <algorithm>
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/tuple.hpp>
#include <exception>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "../include/exceptions.hpp"
#include "../include/problem.hpp"
#include "../include/problems/null_problem.hpp"
#include "../include/serialization.hpp"
#include "../include/types.hpp"
#include "common_utils.hpp"
#include "object_serialization.hpp"

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

template <>
struct prob_inner<bp::object> final: prob_inner_base
{
    // Try to get an attribute from an object. If the call fails,
    // return a def-cted object.
    static bp::object try_attr(const bp::object &o, const char *s)
    {
        bp::object a;
        try {
            a = o.attr(s);
        } catch (...) {
            PyErr_Clear();
        }
        return a;
    }
    // Throw if object does not have a callable attribute.
    static void check_callable_attribute(const bp::object &o, const char *s)
    {
        bp::object a;
        try {
            a = o.attr(s);
        } catch (...) {
            pygmo_throw(PyExc_TypeError,("the mandatory '" + std::string(s) + "()' method is missing").c_str());
        }
        if (!pygmo::callable(a)) {
            pygmo_throw(PyExc_TypeError,("the mandatory '" + std::string(s) + "()' method is not callable").c_str());
        }
    }
    // These are the mandatory methods that must be present.
    void check_construction_object() const
    {
        if (pygmo::isinstance(m_value,pygmo::builtin().attr("type"))) {
            pygmo_throw(PyExc_TypeError,"cannot construct a problem from a type: please use an instance "
                "as construction argument");
        }
        check_callable_attribute(m_value,"fitness");
        check_callable_attribute(m_value,"get_bounds");
    }
    // Just need the def ctor, delete everything else.
    prob_inner() = default;
    prob_inner(const prob_inner &) = delete;
    prob_inner(prob_inner &&) = delete;
    prob_inner &operator=(const prob_inner &) = delete;
    prob_inner &operator=(prob_inner &&) = delete;
    explicit prob_inner(bp::object o):
        // Perform an explicit deep copy of the input object.
        m_value(pygmo::deepcopy(o))
    {
        check_construction_object();
    }
    virtual prob_inner_base *clone() const override final
    {
        // This will make a deep copy using the ctor above.
        return ::new prob_inner(m_value);
    }
    // Mandatory methods.
    virtual vector_double fitness(const vector_double &dv) const override final
    {
        return pygmo::to_vd(m_value.attr("fitness")(pygmo::vd_to_a(dv)));
    }
    virtual std::pair<vector_double,vector_double> get_bounds() const override final
    {
        // First we will try to extract the bounds as a pair of objects.
        bp::tuple tup;
        bp::object obj = m_value.attr("get_bounds")();
        try {
            tup = bp::extract<bp::tuple>(obj);
        } catch (...) {
            pygmo_throw(PyExc_TypeError,"the bounds of the problem must be returned as a tuple");
        }
        // Then we check the tuple size.
        if (len(tup) != 2) {
            pygmo_throw(PyExc_ValueError,("the bounds of the problem must be returned as a tuple of 2 elements, but "
                "the detected tuple size is " + std::to_string(len(tup))).c_str());
        }
        // Finally, we build the pair from the tuple elements.
        return std::make_pair(pygmo::to_vd(tup[0]),pygmo::to_vd(tup[1]));
    }
    // A simple wrapper for the following getters.
    template <typename RetType>
    RetType getter_wrapper(const char *name, const RetType &def_value) const
    {
        auto a = try_attr(m_value,name);
        if (a) {
            return bp::extract<RetType>(a());
        }
        return def_value;
    }
    // Optional methods.
    virtual vector_double::size_type get_nobj() const override final
    {
        return getter_wrapper<vector_double::size_type>("get_nobj",1u);
    }
    virtual vector_double::size_type get_nec() const override final
    {
        return getter_wrapper<vector_double::size_type>("get_nec",0u);
    }
    virtual vector_double::size_type get_nic() const override final
    {
        return getter_wrapper<vector_double::size_type>("get_nic",0u);
    }
    virtual std::string get_name() const override final
    {
        return getter_wrapper<std::string>("get_name",pygmo::str(pygmo::type(m_value)));
    }
    virtual std::string get_extra_info() const override final
    {
        return getter_wrapper<std::string>("get_extra_info",std::string{});
    }
    virtual bool has_gradient() const override final
    {
        const bool hg = try_attr(m_value,"gradient");
        if (hg) {
            // If the problem has a "gradient()" attribute, we further check if
            // it overrides the "has_gradient()" detection. If it does, we call
            // it, otherwise we just return the presence of the "gradient()" method.
            return getter_wrapper<bool>("has_gradient",hg);
        }
        return false;
    }
    virtual vector_double gradient(const vector_double &dv) const override final
    {
        auto a = try_attr(m_value,"gradient");
        if (a) {
            return pygmo::to_vd(a(pygmo::vd_to_a(dv)));
        }
        pygmo_throw(PyExc_RuntimeError,"gradients have been requested but they are not implemented");
    }
    virtual bool has_gradient_sparsity() const override final
    {
        const bool hgs = try_attr(m_value,"gradient_sparsity");
        if (hgs) {
            // If the problem has a "gradient_sparsity()" method, we further
            // check if the override is present. If it is we call it,
            // otherwise we just report the detection of "gradient_sparsity()".
            auto a = try_attr(m_value,"has_gradient_sparsity");
            if (a) {
                return a();
            }
        }
        return hgs;
    }
    virtual sparsity_pattern gradient_sparsity() const override final
    {
        auto a = try_attr(m_value,"gradient_sparsity");
        if (a) {
            return pygmo::to_sp(a());
        }
        // NOTE: this can happen only if somehow the gradient_sparsity() method gets erased *after* the
        // concrete problem has been used in the construction of a problem. This should never happen in normal
        // circumstances, and maybe the error message here could point to some kind of logic error.
        pygmo_throw(PyExc_RuntimeError,"gradient sparsity has been requested but it is not implemented");
    }
    virtual bool has_hessians() const override final
    {
        const bool hh = try_attr(m_value,"hessians");
        if (hh) {
            return getter_wrapper<bool>("has_hessians",hh);
        }
        return false;
    }
    virtual std::vector<vector_double> hessians(const vector_double &dv) const override final
    {
        auto a = try_attr(m_value,"gradient");
        if (a) {
            // Invoke the method, getting out a generic Python object.
            bp::object tmp = a(pygmo::vd_to_a(dv));
            // Check that it is a list.
            if (!pygmo::isinstance(tmp,pygmo::builtin().attr("list"))) {
                pygmo_throw(PyExc_TypeError,"the Hessians must be returned as a list of arrays or lists of doubles");
            }
            // Let's build the return value.
            std::vector<vector_double> retval;
            bp::stl_input_iterator<bp::object> begin(tmp), end;
            std::transform(begin,end,std::back_inserter(retval),[](const auto &o) {
                return pygmo::to_vd(o);
            });
            return retval;
        }
        pygmo_throw(PyExc_RuntimeError,"Hessians have been requested but they are not implemented");
    }
    virtual bool has_hessians_sparsity() const override final
    {
        const bool hhs = try_attr(m_value,"hessians_sparsity");
        if (hhs) {
            auto a = try_attr(m_value,"has_hessians_sparsity");
            if (a) {
                return a();
            }
        }
        return hhs;
    }
    virtual std::vector<sparsity_pattern> hessians_sparsity() const override final
    {
        auto a = try_attr(m_value,"hessians_sparsity");
        if (a) {
            bp::object tmp = a();
            if (!pygmo::isinstance(tmp,pygmo::builtin().attr("list"))) {
                pygmo_throw(PyExc_TypeError,"Hessians sparsities must be returned as a list of sparsity patterns");
            }
            std::vector<sparsity_pattern> retval;
            bp::stl_input_iterator<bp::object> begin(tmp), end;
            std::transform(begin,end,std::back_inserter(retval),[](const auto &o) {
                return pygmo::to_sp(o);
            });
            return retval;
        }
        pygmo_throw(PyExc_RuntimeError,"Hessians sparsities have been requested but they are not implemented");
    }
    virtual void set_seed(unsigned n) override final
    {
        auto a = try_attr(m_value,"set_seed");
        if (a) {
            a(n);
        } else {
            pygmo_throw(PyExc_RuntimeError,"'set_seed()' has been called but it is not implemented");
        }
    }
    virtual bool has_set_seed() const override final
    {
        return getter_wrapper<bool>("has_set_seed",try_attr(m_value,"set_seed"));
    }
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<prob_inner_base>(this),m_value);
    }
    bp::object m_value;
};

}

}

// Register the prob_inner specialised for bp::object.
PAGMO_REGISTER_PROBLEM(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

// Serialization support for the problem class.
struct problem_pickle_suite : bp::pickle_suite
{
    static bp::tuple getinitargs(const pagmo::problem &)
    {
        // For initialization purposes, we use the null problem.
        return bp::make_tuple(pagmo::null_problem{});
    }
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
        return bp::make_tuple(make_bytes(s.data(),boost::numeric_cast<Py_ssize_t>(s.size())));
    }
    static void setstate(pagmo::problem &p, bp::tuple state)
    {
        // Similarly, first we extract a bytes object from the Python state,
        // and then we build a C++ string from it. The string is then used
        // to decerealise the object.
        if (len(state) != 1) {
            pygmo_throw(PyExc_ValueError,"the state tuple must have a single element");
        }
        auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
        if (!ptr) {
            pygmo_throw(PyExc_TypeError,"a bytes object is needed to deserialize a problem");
        }
        const auto size = len(state[0]);
        std::string s(ptr,ptr + size);
        std::istringstream iss;
        iss.str(s);
        {
        cereal::PortableBinaryInputArchive iarchive(iss);
        iarchive(p);
        }
    }
};

// Wrappers for extract.
template <typename Prob, typename T>
inline T problem_cpp_extract(const Prob &p, const T &)
{
    auto ptr = p.template extract<T>();
    if (!ptr) {
        pygmo_throw(PyExc_TypeError,"");
    }
    return *ptr;
}

template <typename Prob>
inline bp::object problem_py_extract(const Prob &p, const bp::object &t)
{
    auto ptr = p.template extract<bp::object>();
    if (!ptr) {
        pygmo_throw(PyExc_TypeError,"could not extract a Python object from the problem");
    }
    if (type(*ptr) != t) {
        pygmo_throw(PyExc_TypeError,("the stored object is not of type " + str(t)).c_str());
    }
    return deepcopy(*ptr);
}

// Wrapper for the fitness function.
inline bp::object fitness_wrapper(const pagmo::problem &p, const bp::object &dv)
{
    return vd_to_a(p.fitness(to_vd(dv)));
}

// Wrapper for the gradient function.
inline bp::object gradient_wrapper(const pagmo::problem &p, const bp::object &dv)
{
    return vd_to_a(p.gradient(to_vd(dv)));
}

// Wrapper for the bounds getter.
inline bp::tuple get_bounds_wrapper(const pagmo::problem &p)
{
    auto retval = p.get_bounds();
    return bp::make_tuple(vd_to_a(retval.first),vd_to_a(retval.second));
}

// Wrapper for gradient sparsity.
inline bp::object gradient_sparsity_wrapper(const pagmo::problem &p)
{
    return sp_to_a(p.gradient_sparsity());
}

// Wrapper for Hessians.
inline bp::list hessians_wrapper(const pagmo::problem &p, const bp::object &dv)
{
    bp::list retval;
    const auto h = p.hessians(to_vd(dv));
    for (const auto &v: h) {
        retval.append(vd_to_a(v));
    }
    return retval;
}

// Wrapper for Hessians sparsity.
inline bp::list hessians_sparsity_wrapper(const pagmo::problem &p)
{
    bp::list retval;
    const auto hs = p.hessians_sparsity();
    for (const auto &sp: hs) {
        retval.append(sp_to_a(sp));
    }
    return retval;
}

}

#endif
