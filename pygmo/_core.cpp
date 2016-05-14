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

#include "../include/exceptions.hpp"
#include "../include/problem.hpp"
#include "../include/problems/hock_schittkowsky_71.hpp"
#include "../include/types.hpp"
#include "pybind11.hpp"

namespace py = pybind11;

namespace pygmo
{

// Perform a deep copy of input object o.
inline py::object deepcopy(py::object o)
{
    return static_cast<py::object>(py::module::import("copy").attr("deepcopy"))(o);
}

// Import and return the builtin module.
inline py::module builtin()
{
#if PY_MAJOR_VERSION < 3
    return py::module::import("__builtin__");
#else
    return py::module::import("builtins");
#endif
}

// Get the type of an object.
inline py::object type(py::object o)
{
    return static_cast<py::object>(builtin().attr("type"))(o);
}

// String representation of an object.
inline py::object str(py::object o)
{
    return static_cast<py::object>(builtin().attr("str"))(o);
}

// Check if type is callable.
inline bool callable(py::object o)
{
    if (!o) {
        return false;
    }
    return static_cast<py::object>(builtin().attr("callable"))(o).cast<bool>();
}

// Convert a vector of doubles into a numpy array.
inline py::array_t<double,py::array::c_style> vd_to_a(const pagmo::vector_double &v)
{
    return py::array_t<double,py::array::c_style>(py::buffer_info(
        // The const_cast should be ok, as there should be no write access into v.
        static_cast<void *>(const_cast<double *>(v.data())),
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        {v.size()},
        {sizeof(double)}
    ));
}

// Convert a numpy array of doubles into a vector of doubles.
inline pagmo::vector_double a_to_vd(py::array_t<double,py::array::c_style> a)
{
    py::buffer_info info = a.request();
    // Check that the input array is actually a 1-dimensional array
    // of doubles.
    if (!info.ptr || info.itemsize != sizeof(double) ||
        info.format != py::format_descriptor<double>::value ||
        info.ndim != 1 || info.shape.size() != 1u || info.strides.size() != 1u ||
        info.strides[0u] != sizeof(double))
    {
        pagmo_throw(std::invalid_argument,"Error creating a vector_double from a NumPy array: the "
            "input array must be a unidimensional array of doubles.");
    }
    return pagmo::vector_double(
        static_cast<double *>(info.ptr),
        static_cast<double *>(info.ptr) + info.shape[0u]
    );
}

// Try converting an arbitrary python object to a vector of doubles. If the input
// is not a list of floats or a 1-dimensional numpy array of floats, an error will be thrown.
inline pagmo::vector_double to_vd(py::object o)
{
    py::module nm = py::module::import("numpy");
    py::object ndarray = nm.attr("ndarray");
    py::object isinstance = builtin().attr("isinstance");
    py::object list = builtin().attr("list");
    if (isinstance(o,ndarray).cast<bool>()) {
        return a_to_vd(o);
    } else if (isinstance(o,list).cast<bool>()) {
        try {
            return o.cast<pagmo::vector_double>();
        } catch (const py::cast_error &) {
            // This means that pybind11 was not able to cast o to a vector of doubles.
            // We will raise an appropriate error below.
        }
    }
    pagmo_throw(std::logic_error,"Cannot convert the type '" + str(type(o)).cast<std::string>() + "' to a "
        "C++ vector of doubles. Only lists of floats and NumPy arrays of floats "
        "are supported.");
}

}

namespace pagmo
{

namespace detail
{

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
            pagmo_throw(std::logic_error,"the '" + std::string(s) + "' method is missing or "
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
        return attr(m_value,"get_nobj")().cast<vector_double::size_type>();
    }
    virtual std::pair<vector_double,vector_double> get_bounds() const override final
    {
        return attr(m_value,"get_bounds")()
            .cast<std::pair<vector_double,vector_double>>();
    }
    virtual vector_double::size_type get_nec() const override final
    {
        auto a = attr(m_value,"get_nec");
        if (pygmo::callable(a)) {
            return a().cast<vector_double::size_type>();
        }
        return 0u;
    }
    virtual vector_double::size_type get_nic() const override final
    {
        auto a = attr(m_value,"get_nic");
        if (pygmo::callable(a)) {
            return a().cast<vector_double::size_type>();
        }
        return 0u;
    }
    virtual std::string get_name() const override final
    {
        auto a = attr(m_value,"get_name");
        if (pygmo::callable(a)) {
            return a().cast<std::string>();
        }
        return pygmo::str(pygmo::type(m_value)).cast<std::string>();
    }
    virtual std::string get_extra_info() const override final
    {
        auto a = attr(m_value,"get_extra_info");
        if (pygmo::callable(a)) {
            return a().cast<std::string>();
        }
        return "";
    }
    virtual bool has_gradient() const override final
    {
        return pygmo::callable(attr(m_value,"gradient"));
    }
    virtual vector_double gradient(const vector_double &x) const override final
    {
        auto a = attr(m_value,"gradient");
        if (pygmo::callable(a)) {
            return a(x).cast<vector_double>();
        }
        pagmo_throw(std::logic_error,"Gradients have been requested but they are not implemented or not implemented correctly.");
    }
    virtual bool has_gradient_sparsity() const override final
    {
        // If the concrete problem implements has_gradient_sparsity use it,
        // otherwise check if the gradient_sparsity method exists.
        auto a = attr(m_value,"has_gradient_sparsity");
        if (pygmo::callable(a)) {
            return a().cast<bool>();
        }
        return pygmo::callable(attr(m_value,"gradient_sparsity"));
    }
    virtual sparsity_pattern gradient_sparsity() const override final
    {
        auto a = attr(m_value,"gradient_sparsity");
        if (pygmo::callable(a)) {
            return a().cast<sparsity_pattern>();
        }
        pagmo_throw(std::logic_error,"Gradient sparsity has been requested but it is not implemented or not implemented correctly.");
    }
    virtual bool has_hessians() const override final
    {
        return pygmo::callable(attr(m_value,"hessians"));
    }
    virtual std::vector<vector_double> hessians(const vector_double &x) const override final
    {
        auto a = attr(m_value,"hessians");
        if (pygmo::callable(a)) {
            return a(x).cast<std::vector<vector_double>>();
        }
        pagmo_throw(std::logic_error,"Hessians have been requested but they are not implemented or not implemented correctly.");
    }
    virtual bool has_hessians_sparsity() const override final
    {
        auto a = attr(m_value,"has_hessians_sparsity");
        if (pygmo::callable(a)) {
            return a().cast<bool>();
        }
        return pygmo::callable(attr(m_value,"hessians_sparsity"));
    }
    virtual std::vector<sparsity_pattern> hessians_sparsity() const override final
    {
        auto a = attr(m_value,"hessians_sparsity");
        if (pygmo::callable(a)) {
            return a().cast<std::vector<sparsity_pattern>>();
        }
        pagmo_throw(std::logic_error,"Hessians sparsities have been requested but they are not implemented or not implemented correctly.");
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
            return a().cast<bool>();
        }
        return pygmo::callable(attr(m_value,"set_seed"));
    }
    py::object m_value;
};

}

}

using namespace pagmo;

PYBIND11_PLUGIN(_core)
{
    py::module m("_core", "PyGMO's core module");

    py::class_<problem> problem_class(m,"problem");

    // Expose the generic problem interface.
    problem_class.def(py::init<const problem &>())
        .def("fitness",[](const problem &p, py::array_t<double,py::array::c_style> dv) {
            return pygmo::vd_to_a(p.fitness(pygmo::a_to_vd(dv)));
        },"Fitness.", py::arg("dv"))
        .def("gradient",&problem::gradient)
        .def("has_gradient",&problem::has_gradient)
        .def("gradient_sparsity",&problem::gradient_sparsity)
        .def("has_gradient_sparsity",&problem::has_gradient_sparsity)
        .def("hessians",&problem::hessians)
        .def("has_hessians",&problem::has_hessians)
        .def("hessians_sparsity",&problem::hessians_sparsity)
        .def("has_hessians_sparsity",&problem::has_hessians_sparsity)
        .def("get_nobj",&problem::get_nobj)
        .def("get_nx",&problem::get_nx)
        .def("get_nf",&problem::get_nf)
        .def("get_bounds",&problem::get_bounds)
        .def("get_nec",&problem::get_nec)
        .def("get_nic",&problem::get_nic)
        .def("get_nc",&problem::get_nc)
        .def("get_fevals",&problem::get_fevals)
        .def("get_gevals",&problem::get_gevals)
        .def("get_hevals",&problem::get_hevals)
        .def("get_gs_dim",&problem::get_gs_dim)
        .def("get_hs_dim",&problem::get_hs_dim)
        .def("set_seed",&problem::set_seed)
        .def("has_set_seed",&problem::has_set_seed)
        .def("is_stochastic",&problem::is_stochastic)
        .def("get_name",&problem::get_name)
        .def("get_extra_info",&problem::get_extra_info)
        .def("__repr__",[](const problem &p) {
            std::stringstream oss;
            oss << p;
            return oss.str();
        });

    py::class_<hock_schittkowsky_71> hs71(m,"hock_schittkowsky_71");
    hs71.def(py::init<>());

    problem_class.def(py::init<hock_schittkowsky_71>());
    problem_class.def("_extract",[](const problem &p, const hock_schittkowsky_71 &) {
        auto ptr = p.extract<hock_schittkowsky_71>();
        if (!ptr) {
            pagmo_throw(std::runtime_error,std::string("cannot extract an instance of type '") +
                typeid(hock_schittkowsky_71).name() + "'");
        }
        return hock_schittkowsky_71(*ptr);
    });

    // This needs to go last, as it needs to have the lowest precedence among all ctors.
    problem_class.def(py::init<py::object>());
    problem_class.def("_extract",[](const problem &p, py::object o) {
        auto ptr = p.extract<py::object>();
        if (!ptr || pygmo::type(*ptr) != pygmo::type(o)) {
            pagmo_throw(std::runtime_error,"cannot extract an instance of type '" +
                pygmo::str(pygmo::type(o)).cast<std::string>() + "'");
        }
        return pygmo::deepcopy(*ptr);
    });

    return m.ptr();
}
