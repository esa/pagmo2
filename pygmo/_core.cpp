// NOTE: the order of inclusion in the first two items here is forced by these two issues:
// http://mail.python.org/pipermail/python-list/2004-March/907592.html
// http://mail.python.org/pipermail/new-bugs-announce/2011-March/010395.html
#if defined(_WIN32)
#include <cmath>
#include <Python.h>
#else
#include <Python.h>
#include <cmath>
#endif

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpedantic"
    #pragma GCC diagnostic ignored "-Wshadow"
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wdeprecated"
#endif

#include "../include/external/pybind11/include/pybind11/pybind11.h"
#include "../include/external/pybind11/include/pybind11/stl.h"

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#include "../include/exceptions.hpp"
#include "../include/problem.hpp"
#include "../include/types.hpp"

namespace py = pybind11;

namespace pagmo
{

namespace detail
{

template <>
struct prob_inner<py::object>: prob_inner_base
{
    void check_construction_object() const
    {
        auto attr = static_cast<py::object>(m_value.attr("fitness"));
        if (!attr) {
            pagmo_throw(std::invalid_argument,"the 'fitness()' method is missing");
        }
        attr = static_cast<py::object>(m_value.attr("get_nobj"));
        if (!attr) {
            pagmo_throw(std::invalid_argument,"the 'get_nobj()' method is missing");
        }
        attr = static_cast<py::object>(m_value.attr("get_bounds"));
        if (!attr) {
            pagmo_throw(std::invalid_argument,"the 'get_bounds()' method is missing");
        }
    }
    explicit prob_inner(py::object &&x):m_value(std::move(x))
    {
        check_construction_object();
    }
    // TODO need to understand here: do we need deepc copy? What about the generic ctor.
    explicit prob_inner(const py::object &x):m_value(x)
    {
        check_construction_object();
    }
    virtual prob_inner_base *clone() const override final
    {
        return ::new prob_inner<py::object>(m_value);
    }
    // Main methods.
    virtual vector_double fitness(const vector_double &dv) const override final
    {
        return static_cast<py::object>(m_value.attr("fitness")).call(dv).cast<vector_double>();
    }
    virtual vector_double::size_type get_nobj() const override final
    {
        return static_cast<py::object>(m_value.attr("get_nobj")).call().cast<vector_double::size_type>();
    }
    virtual std::pair<vector_double,vector_double> get_bounds() const override final
    {
        return static_cast<py::object>(m_value.attr("get_bounds")).call()
            .cast<std::pair<vector_double,vector_double>>();
    }
    virtual vector_double::size_type get_nec() const override final
    {
        auto attr1 = static_cast<py::object>(m_value.attr("get_nec"));
        auto attr2 = static_cast<py::object>(m_value.attr("get_nic"));
        if (attr1 && attr2) {
            return attr1.call().cast<vector_double::size_type>();
        }
        return 0u;
    }
    virtual vector_double::size_type get_nic() const override final
    {
        auto attr1 = static_cast<py::object>(m_value.attr("get_nec"));
        auto attr2 = static_cast<py::object>(m_value.attr("get_nic"));
        if (attr1 && attr2) {
            return attr2.call().cast<vector_double::size_type>();
        }
        return 0u;
    }
    virtual std::string get_name() const override final
    {
        auto attr = static_cast<py::object>(m_value.attr("get_name"));
        if (attr) {
            return attr.call().cast<std::string>();
        }
#if PY_MAJOR_VERSION < 3
        auto m = py::module::import("__builtin__");
#else
        auto m = py::module::import("builtins");
#endif
        auto type = static_cast<py::object>(m.attr("type")),
            str = static_cast<py::object>(m.attr("str"));
        return str.call(type.call(m_value)).cast<std::string>();
    }
    virtual std::string get_extra_info() const override final
    {
        auto attr = static_cast<py::object>(m_value.attr("get_extra_info"));
        if (attr) {
            return attr.call().cast<std::string>();
        }
        return "";
    }
    virtual bool has_gradient() const override final
    {
        return (static_cast<py::object>(m_value.attr("gradient")));
    }
    virtual vector_double gradient(const vector_double &x) const override final
    {
        auto attr = static_cast<py::object>(m_value.attr("gradient"));
        if (attr) {
            return attr.call(x).cast<vector_double>();
        }
        pagmo_throw(std::logic_error,"Gradients have been requested but they are not implemented or not implemented correctly.");
    }
    virtual bool has_gradient_sparsity() const override final
    {
        return (static_cast<py::object>(m_value.attr("gradient_sparsity")));
    }
    virtual sparsity_pattern gradient_sparsity() const override final
    {
        auto attr = static_cast<py::object>(m_value.attr("gradient_sparsity"));
        if (attr) {
            return attr.call().cast<sparsity_pattern>();
        }
        pagmo_throw(std::logic_error,"Sparsity pattern has been requested but it is not implemented or not implemented correctly.");
    }
    virtual bool has_hessians() const override final
    {
        return (static_cast<py::object>(m_value.attr("hessians")));
    }
    virtual std::vector<vector_double> hessians(const vector_double &x) const override final
    {
        auto attr = static_cast<py::object>(m_value.attr("hessians"));
        if (attr) {
            return attr.call(x).cast<std::vector<vector_double>>();
        }
        pagmo_throw(std::logic_error,"Hessians have been requested but they are not implemented or not implemented correctly.");
    }
    virtual bool has_hessians_sparsity() const override final
    {
        return (static_cast<py::object>(m_value.attr("hessians_sparsity")));
    }
    virtual std::vector<sparsity_pattern> hessians_sparsity() const override final
    {
        auto attr = static_cast<py::object>(m_value.attr("hessians_sparsity"));
        if (attr) {
            return attr.call().cast<std::vector<sparsity_pattern>>();
        }
        pagmo_throw(std::logic_error,"Hessians sparsity patterns have been requested but they are not implemented or not implemented correctly.");
    }

    // Serialization.
//     template <typename Archive>
//     void serialize(Archive &ar)
//     {
//         ar(cereal::base_class<prob_inner_base>(this),m_value);
//     }
    py::object m_value;
};

}

}

using namespace pagmo;


PYBIND11_PLUGIN(_core)
{
    py::module m("_core", "PyGMO's core module");

    py::class_<problem>(m,"problem")
        .def(py::init<py::object>())
        .def("fitness",&problem::fitness)
        .def("get_bounds",&problem::get_bounds)
        .def("get_fevals",&problem::get_fevals)
        .def("__repr__",[](const problem &p) {
            std::stringstream oss;
            oss << p;
            return oss.str();
        });

    return m.ptr();
}
