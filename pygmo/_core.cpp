#include <memory>
#include <stdexcept>
#include <utility>

#include "../include/exceptions.hpp"
#include "../include/external/pybind11/include/pybind11/numpy.h"
#include "../include/external/pybind11/include/pybind11/pybind11.h"
#include "../include/external/pybind11/include/pybind11/stl.h"
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
        py::object attr = m_value.attr("fitness");
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
    virtual prob_inner_base *clone() const override
    {
        return ::new prob_inner<py::object>(m_value);
    }
    // Main methods.
    virtual fitness_vector fitness(const decision_vector &dv) override
    {
        py::object attr = m_value.attr("fitness");
        return attr.call(dv).cast<fitness_vector>();
    }
    virtual fitness_vector::size_type get_nf() const override
    {
        return 1u;
    }
    virtual decision_vector::size_type get_n() const override
    {
        return 1u;
    }
    virtual std::pair<decision_vector,decision_vector> get_bounds() const override
    {
        return {{0},{1}};
    }
    virtual decision_vector::size_type get_nec() const override
    {
        return 0u;
    }
    virtual decision_vector::size_type get_nic() const override
    {
        return 0u;
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
        .def("get_bounds",&problem::get_bounds);

    return m.ptr();
}
