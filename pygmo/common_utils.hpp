#ifndef PYGMO_COMMON_UTILS_HPP
#define PYGMO_COMMON_UTILS_HPP

#include "python_includes.hpp"

#include <algorithm>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/handle.hpp>
#include <boost/python/import.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>
#include <stdexcept>
#include <string>

#include "../include/exceptions.hpp"
#include "../include/types.hpp"
#include "numpy.hpp"

namespace pygmo
{

namespace bp = boost::python;

// Perform a deep copy of input object o.
inline bp::object deepcopy(bp::object o)
{
    return bp::import("copy").attr("deepcopy")(o);
}

// Import and return the builtin module.
inline bp::object builtin()
{
#if PY_MAJOR_VERSION < 3
    return bp::import("__builtin__");
#else
    return bp::import("builtins");
#endif
}

// Get the type of an object.
inline bp::object type(bp::object o)
{
    return builtin().attr("type")(o);
}

// String representation of an object.
inline bp::object str(bp::object o)
{
    return builtin().attr("str")(o);
}

// Check if type is callable.
inline bool callable(bp::object o)
{
    if (!o) {
        return false;
    }
    return bp::extract<bool>(builtin().attr("callable")(o));
}

// Convert a vector of doubles into a numpy array.
inline bp::object vd_to_a(const pagmo::vector_double &v)
{
    // The dimensions of the array to be created.
    npy_intp dims[] = {boost::numeric_cast<npy_intp>(v.size())};
    // Attempt creating the array.
    PyObject *ret = PyArray_SimpleNew(1,dims,NPY_DOUBLE);
    if (!ret) {
        pagmo_throw(std::runtime_error,"couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    // Copy over the data.
    std::copy(v.begin(),v.end(),static_cast<double *>(PyArray_DATA((PyArrayObject *)(ret))));
    // Hand over to boost python.
    return bp::object(bp::handle<>(ret));
}

// isinstance wrapper.
inline bool isinstance(bp::object o, bp::object t)
{
    return bp::extract<bool>(builtin().attr("isinstance")(o,t));
}

// Convert a numpy array to a vector_double.
inline pagmo::vector_double a_to_vd(PyArrayObject *o)
{
    using size_type = pagmo::vector_double::size_type;
    if (!PyArray_ISCARRAY_RO(o)) {
        pagmo_throw(std::runtime_error,"cannot convert NumPy array to a vector of doubles: "
         "data must be C-style contiguous, aligned, and in machine byte-order");
    }
    if (PyArray_NDIM(o) != 1) {
        pagmo_throw(std::runtime_error,"cannot convert NumPy array to a vector of doubles: "
         "the array must be unidimensional");
    }
    if (PyArray_TYPE(o) != NPY_DOUBLE) {
        pagmo_throw(std::runtime_error,"cannot convert NumPy array to a vector of doubles: "
         "the scalar type must be 'double'");
    }
    if (PyArray_STRIDES(o)[0] != sizeof(double)) {
        pagmo_throw(std::runtime_error,"cannot convert NumPy array to a vector of doubles: "
         "the stride value must be " + std::to_string(sizeof(double)));
    }
    if (PyArray_ITEMSIZE(o) != sizeof(double)) {
        pagmo_throw(std::runtime_error,"cannot convert NumPy array to a vector of doubles: "
         "the size of the scalar type must be " + std::to_string(sizeof(double)));
    }
    // NOTE: not sure if this special casing is needed. We make sure
    // the array contains something in order to avoid messing around
    // with a potentially null pointer in the array.
    const auto size = boost::numeric_cast<size_type>(PyArray_SHAPE(o)[0]);
    if (size) {
        auto data = static_cast<double *>(PyArray_DATA(o));
        return pagmo::vector_double(data,data + size);
    }
    return pagmo::vector_double{};
}

// Convert an arbitrary python object to a vector_double.
inline pagmo::vector_double to_vd(bp::object o)
{
    bp::object l = builtin().attr("list");
    bp::object a = bp::import("numpy").attr("ndarray");
    if (isinstance(o,l)) {
        bp::stl_input_iterator<double> begin(o), end;
        return pagmo::vector_double(begin,end);
    } else if (isinstance(o,a)) {
        return a_to_vd((PyArrayObject *)(o.ptr()));
    }
    pagmo_throw(std::runtime_error,"cannot convert the type '" + static_cast<std::string>(bp::extract<std::string>(str(type(o)))) + "' to a "
        "vector of doubles: only lists of objects convertible to doubles and NumPy arrays of doubles "
        "are supported");
}

// Convert a sparsity pattern into a numpy array.
inline bp::object sp_to_a(const pagmo::sparsity_pattern &s)
{
    npy_intp dims[] = {boost::numeric_cast<npy_intp>(s.size()),2};
    PyObject *ret = PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    if (!ret) {
        pagmo_throw(std::runtime_error,"couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    // TODO zero size here and above?
    auto data = static_cast<double *>(PyArray_DATA((PyArrayObject *)(ret)));
    for (decltype(s.size()) i = 0u; i < s.size(); ++i) {
        // TODO range checks?
        *(data + 2u*i) = s[i].first;
        *(data + 2u*i + 1u) = s[i].second;
    }
    // Hand over to boost python.
    return bp::object(bp::handle<>(ret));
}

#if 0

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

#endif

}

#endif
