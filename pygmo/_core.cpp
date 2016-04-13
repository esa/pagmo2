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
//     prob_inner() = default;
    void check_construction_object() const
    {
        auto attr = static_cast<py::object>(m_value.attr("fitness"));
        if (!attr) {
            pagmo_throw(std::invalid_argument,"the 'fitness()' method is missing");
        }
    }
    explicit prob_inner(py::object &&x):m_value(std::move(x))
    {
        check_construction_object();
    }
    explicit prob_inner(const py::object &x):m_value(x)
    {
        check_construction_object();
    }
    virtual prob_inner_base *clone() const override final
    {
        return ::new prob_inner<py::object>(m_value);
    }
    // Main methods.
    virtual fitness_vector fitness(const decision_vector &dv) override final
    {
        py::object attr = m_value.attr("fitness");
        return attr.call(dv).cast<fitness_vector>();
    }
    virtual fitness_vector::size_type get_nf() const override final
    {
        return 1u;
    }
    virtual decision_vector::size_type get_n() const override final
    {
        return 1u;
    }
    virtual std::pair<decision_vector,decision_vector> get_bounds() const override final
    {
        return {{0},{1}};
    }
    virtual decision_vector::size_type get_nec() const override final
    {
        return 0u;
    }
    virtual decision_vector::size_type get_nic() const override final
    {
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
        .def("fitness_vector",&problem::fitness)
        .def("get_bounds",&problem::get_bounds)
        .def("get_fevals",&problem::get_fevals)
        .def("__repr__",[](const problem &p) {
            std::stringstream oss;
            oss << p;
            return oss.str();
        });

    return m.ptr();
}
