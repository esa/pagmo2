#ifndef PYGMO_COMMON_UTILS_HPP
#define PYGMO_COMMON_UTILS_HPP

#include <stdexcept>
#include <string>

#include "../include/exceptions.hpp"
#include "../include/types.hpp"
#include "pybind11.hpp"

namespace pygmo
{

namespace py = pybind11;

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
        pagmo_throw(std::invalid_argument,"error creating a vector of doubles from a NumPy array: the "
            "input array must be a unidimensional array of doubles");
    }
    return pagmo::vector_double(
        static_cast<double *>(info.ptr),
        static_cast<double *>(info.ptr) + info.shape[0u]
    );
}

// Convert a sparsity pattern into a numpy array.
inline py::array_t<pagmo::vector_double::size_type,py::array::c_style> sp_to_a(const pagmo::sparsity_pattern &s)
{
    using size_type = pagmo::vector_double::size_type;
    // Copy the sparsity pattern to a temporary buffer.
    std::vector<size_type> tmp;
    for (const auto &p: s) {
        tmp.push_back(p.first);
        tmp.push_back(p.second);
    }
    return py::array_t<size_type,py::array::c_style>(py::buffer_info(
        static_cast<void *>(const_cast<size_type *>(tmp.data())),
        sizeof(size_type),
        py::format_descriptor<size_type>::value,
        2,
        {s.size(),2u},
        {sizeof(size_type) * 2u, sizeof(size_type)}
    ));
}

// Convert a numpy array of vector_double::size_type into a sparsity pattern.
inline pagmo::sparsity_pattern a_to_sp(py::array_t<pagmo::vector_double::size_type,py::array::c_style> a)
{
    using size_type = pagmo::vector_double::size_type;
    py::buffer_info info = a.request();
    if (!info.ptr || info.itemsize != sizeof(size_type) ||
        info.format != py::format_descriptor<size_type>::value ||
        info.ndim != 2 || info.shape.size() != 2u || info.strides.size() != 2u ||
        info.strides[0u] != sizeof(size_type) * 2u || info.strides[1u] != sizeof(size_type))
    {
        pagmo_throw(std::invalid_argument,"error creating a sparsity pattern from a NumPy array: the "
            "input array must be a Nx2 array of pagmo::vector_double::size_type");
    }
    pagmo::sparsity_pattern retval;
    auto l = info.shape[0u];
    for (decltype(l) i = 0; i < l; ++i) {
        retval.emplace_back(*(static_cast<size_type *>(info.ptr) + 2u * i),
            *(static_cast<size_type *>(info.ptr) + 2u * i + 1u));
    }
    return retval;
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
    pagmo_throw(std::logic_error,"cannot convert the type '" + str(type(o)).cast<std::string>() + "' to a "
        "vector of doubles: only lists of floats and NumPy arrays of floats "
        "are supported");
}

// Try converting a python object to a sparsity pattern.
inline pagmo::sparsity_pattern to_sp(py::object o)
{
    py::module nm = py::module::import("numpy");
    py::object ndarray = nm.attr("ndarray");
    py::object isinstance = builtin().attr("isinstance");
    py::object list = builtin().attr("list");
    if (isinstance(o,ndarray).cast<bool>()) {
        return a_to_sp(o);
    } else if (isinstance(o,list).cast<bool>()) {
        try {
            return o.cast<pagmo::sparsity_pattern>();
        } catch (const py::cast_error &) {}
    }
    pagmo_throw(std::logic_error,"cannot convert the type '" + str(type(o)).cast<std::string>() + "' to a "
        "sparsity pattern: only lists of 2-tuples of pagmo::vector_double::size_type and NumPy arrays of "
        "pagmo::vector_double::size_type are supported");
}

}

#endif
