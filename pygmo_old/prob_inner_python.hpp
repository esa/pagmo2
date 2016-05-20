#ifndef PYGMO_PROB_INNER_PYTHON_HPP
#define PYGMO_PROB_INNER_PYTHON_HPP

#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "../include/exceptions.hpp"
#include "../include/problem.hpp"
#include "../include/types.hpp"
#include "common_utils.hpp"
#include "pybind11.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

template <>
struct prob_inner<py::object> final: prob_inner_base
{
    // Return instance attribute as a py::object.
    static py::object attr(py::object o, const char *s)
    {
        return o.attr(s);
    }
    // Throw if object does not have a callable attribute.
    static void check_callable_attribute(py::object o, const char *s)
    {
        if (!pygmo::callable(attr(o,s))) {
            pagmo_throw(std::logic_error,"the '" + std::string(s) + "()' method is missing or "
                "it is not callable");
        }
    }
    // These are the mandatory methods that must be present.
    void check_construction_object() const
    {
        check_callable_attribute(m_value,"fitness");
        check_callable_attribute(m_value,"get_nobj");
        check_callable_attribute(m_value,"get_bounds");
    }
    // Just need the def ctor, delete everything else.
    prob_inner() = default;
    prob_inner(const prob_inner &) = delete;
    prob_inner(prob_inner &&) = delete;
    prob_inner &operator=(const prob_inner &) = delete;
    prob_inner &operator=(prob_inner &&) = delete;
    explicit prob_inner(py::object o):
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
    // Main methods.
    virtual vector_double fitness(const vector_double &dv) const override final
    {
        return pygmo::to_vd(attr(m_value,"fitness")(pygmo::vd_to_a(dv)));
    }
    virtual vector_double::size_type get_nobj() const override final
    {
        try {
            return attr(m_value,"get_nobj")().cast<vector_double::size_type>();
        } catch (const py::cast_error &) {
            // NOTE: here we are catching the cast_error for 2 reasons:
            // - to provide a more helpful error message,
            // - to work around a peculiar pybind11 behaviour. Basically, it seems
            //   like the cast_error is caught somewhere inside pybind11 when raised,
            //   and the final error message produced on the Python prompt when this fails
            //   from the problem constructor is misleading as a consquence (it reads as if
            //   as suitable constructor hadn't been provided). We adopt the same pattern
            //   throughout alll these methods.
            pagmo_throw(std::logic_error,"could not convert the output of the 'get_nobj()' method to an integral value");
        }
    }
    virtual std::pair<vector_double,vector_double> get_bounds() const override final
    {
        // First we will try to extract the bounds as a pair of objects.
        std::pair<py::object,py::object> p;
        try {
            p = attr(m_value,"get_bounds")()
                .cast<std::pair<py::object,py::object>>();
        } catch (const py::cast_error &) {
            pagmo_throw(std::logic_error,"the bounds must be returned as a tuple of 2 arrays");
        }
        // Then we try to construct vectors of doubles from the objects in the tuple.
        return std::make_pair(pygmo::to_vd(p.first),pygmo::to_vd(p.second));
    }
    virtual vector_double::size_type get_nec() const override final
    {
        auto a = attr(m_value,"get_nec");
        if (pygmo::callable(a)) {
            try {
                return a().cast<vector_double::size_type>();
            } catch (const py::cast_error &) {
                pagmo_throw(std::logic_error,"could not convert the output of the 'get_nec()' method to an integral value");
            }
        }
        return 0u;
    }
    virtual vector_double::size_type get_nic() const override final
    {
        auto a = attr(m_value,"get_nic");
        if (pygmo::callable(a)) {
            try {
                return a().cast<vector_double::size_type>();
            } catch (const py::cast_error &) {
                pagmo_throw(std::logic_error,"could not convert the output of the 'get_nic()' method to an integral value");
            }
        }
        return 0u;
    }
    virtual std::string get_name() const override final
    {
        auto a = attr(m_value,"get_name");
        if (pygmo::callable(a)) {
            try {
                return a().cast<std::string>();
            } catch (const py::cast_error &) {
                pagmo_throw(std::logic_error,"could not convert the output of the 'get_name()' method to a string");
            }
        }
        return pygmo::str(pygmo::type(m_value)).cast<std::string>();
    }
    virtual std::string get_extra_info() const override final
    {
        auto a = attr(m_value,"get_extra_info");
        if (pygmo::callable(a)) {
            try {
                return a().cast<std::string>();
            } catch (const py::cast_error &) {
                pagmo_throw(std::logic_error,"could not convert the output of the 'get_extra_info()' method to a string");
            }
        }
        return "";
    }
    virtual bool has_gradient() const override final
    {
        return pygmo::callable(attr(m_value,"gradient"));
    }
    virtual vector_double gradient(const vector_double &dv) const override final
    {
        auto a = attr(m_value,"gradient");
        if (pygmo::callable(a)) {
            return pygmo::to_vd(a(pygmo::vd_to_a(dv)));
        }
        pagmo_throw(std::logic_error,"gradients have been requested but they are not implemented or not implemented correctly");
    }
    virtual bool has_gradient_sparsity() const override final
    {
        return pygmo::callable(attr(m_value,"gradient_sparsity"));
    }
    virtual sparsity_pattern gradient_sparsity() const override final
    {
        auto a = attr(m_value,"gradient_sparsity");
        if (pygmo::callable(a)) {
            return pygmo::to_sp(a());
        }
        pagmo_throw(std::logic_error,"gradient sparsity has been requested but it is not implemented or not implemented correctly");
    }
    virtual bool has_hessians() const override final
    {
        return pygmo::callable(attr(m_value,"hessians"));
    }
    virtual std::vector<vector_double> hessians(const vector_double &dv) const override final
    {
        auto a = attr(m_value,"hessians");
        if (pygmo::callable(a)) {
            // First let's try to extract a vector of objects.
            std::vector<py::object> tmp;
            try {
                tmp = a(pygmo::vd_to_a(dv)).cast<std::vector<py::object>>();
            } catch (const py::cast_error &) {
                pagmo_throw(std::logic_error,"Hessians must be returned as a list of arrays");
            }
            // Now we build the return value.
            std::vector<vector_double> retval;
            std::transform(tmp.begin(),tmp.end(),std::back_inserter(retval),[](const auto &o) {
                return pygmo::to_vd(o);
            });
            return retval;
        }
        pagmo_throw(std::logic_error,"Hessians have been requested but they are not implemented or not implemented correctly");
    }
    virtual bool has_hessians_sparsity() const override final
    {
        return pygmo::callable(attr(m_value,"hessians_sparsity"));
    }
    virtual std::vector<sparsity_pattern> hessians_sparsity() const override final
    {
        auto a = attr(m_value,"hessians_sparsity");
        if (pygmo::callable(a)) {
            std::vector<py::object> tmp;
            try {
                tmp = a().cast<std::vector<py::object>>();
            } catch (const py::cast_error &) {
                pagmo_throw(std::logic_error,"Hessians sparsities must be returned as a list of sparsity patterns");
            }
            std::vector<sparsity_pattern> retval;
            std::transform(tmp.begin(),tmp.end(),std::back_inserter(retval),[](const auto &o) {
                return pygmo::to_sp(o);
            });
            return retval;
        }
        pagmo_throw(std::logic_error,"Hessians sparsities have been requested but they are not implemented or not implemented correctly");
    }
    virtual void set_seed(unsigned n) override final
    {
        auto a = attr(m_value,"set_seed");
        if (pygmo::callable(a)) {
            a(n);
        } else {
            pagmo_throw(std::logic_error,"'set_seed()' has been called but it is not implemented or not implemented correctly");
        }
    }
    virtual bool has_set_seed() const override final
    {
        auto a = attr(m_value,"has_set_seed");
        if (pygmo::callable(a)) {
            try {
                return a().cast<bool>();
            } catch (const py::cast_error &) {
                pagmo_throw(std::logic_error,"could not convert the output of the 'has_set_seed()' method to a boolean");
            }
        }
        return pygmo::callable(attr(m_value,"set_seed"));
    }
    py::object m_value;
};

}

}

#endif
