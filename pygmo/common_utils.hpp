#ifndef PYGMO_COMMON_UTILS_HPP
#define PYGMO_COMMON_UTILS_HPP

#include "python_includes.hpp"

#include <algorithm>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/handle.hpp>
#include <boost/python/import.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/tuple.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "../include/exceptions.hpp"
#include "../include/types.hpp"
#include "numpy.hpp"

// A throwing macro similar to pagmo_throw, only for Python. This will set the global
// error string of Python to "msg", the exception type to "type", and then invoke the Boost
// Python function to raise the Python exception.
#define pygmo_throw(type,msg) \
::PyErr_SetString(type,msg); \
bp::throw_error_already_set(); \
throw

namespace pygmo
{

namespace bp = boost::python;

// Map C++ types to NPY_ types.
template <typename T>
struct cpp_npy {};

#define PYGMO_CPP_NPY(from,to) \
template <> \
struct cpp_npy<from> \
{ \
    static constexpr auto value = to; \
};

// We only need integral types at the moment.
PYGMO_CPP_NPY(unsigned char,NPY_UBYTE)
PYGMO_CPP_NPY(unsigned short,NPY_USHORT)
PYGMO_CPP_NPY(unsigned,NPY_UINT)
PYGMO_CPP_NPY(unsigned long,NPY_ULONG)
PYGMO_CPP_NPY(unsigned long long,NPY_ULONGLONG)
PYGMO_CPP_NPY(signed char,NPY_BYTE)
PYGMO_CPP_NPY(short,NPY_SHORT)
PYGMO_CPP_NPY(int,NPY_INT)
PYGMO_CPP_NPY(long,NPY_LONG)
PYGMO_CPP_NPY(long long,NPY_LONGLONG)

#undef PYGMO_CPP_NPY

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
inline std::string str(bp::object o)
{
    return bp::extract<std::string>(builtin().attr("str")(o));
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
        pygmo_throw(PyExc_RuntimeError,"couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    if (v.size()) {
        // Copy over the data.
        std::copy(v.begin(),v.end(),static_cast<double *>(PyArray_DATA((PyArrayObject *)(ret))));
    }
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
        pygmo_throw(PyExc_RuntimeError,"cannot convert NumPy array to a vector of doubles: "
         "data must be C-style contiguous, aligned, and in machine byte-order");
    }
    if (PyArray_NDIM(o) != 1) {
        pygmo_throw(PyExc_ValueError,"cannot convert NumPy array to a vector of doubles: "
         "the array must be unidimensional");
    }
    if (PyArray_TYPE(o) != NPY_DOUBLE) {
        pygmo_throw(PyExc_TypeError,"cannot convert NumPy array to a vector of doubles: "
         "the scalar type must be 'double'");
    }
    if (PyArray_STRIDES(o)[0] != sizeof(double)) {
        pygmo_throw(PyExc_RuntimeError,("cannot convert NumPy array to a vector of doubles: "
         "the stride value must be " + std::to_string(sizeof(double))).c_str());
    }
    if (PyArray_ITEMSIZE(o) != sizeof(double)) {
        pygmo_throw(PyExc_RuntimeError,("cannot convert NumPy array to a vector of doubles: "
         "the size of the scalar type must be " + std::to_string(sizeof(double))).c_str());
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
    pygmo_throw(PyExc_TypeError,("cannot convert the type '" + str(type(o)) + "' to a "
        "vector of doubles: only lists of doubles and NumPy arrays of doubles "
        "are supported").c_str());
}

// Convert a sparsity pattern into a numpy array.
inline bp::object sp_to_a(const pagmo::sparsity_pattern &s)
{
    // The unsigned integral type that is used in the sparsity pattern.
    using size_type = pagmo::vector_double::size_type;
    npy_intp dims[] = {boost::numeric_cast<npy_intp>(s.size()),2};
    PyObject *ret = PyArray_SimpleNew(2,dims,cpp_npy<size_type>::value);
    if (!ret) {
        pygmo_throw(PyExc_RuntimeError,"couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    // NOTE: same as above, avoid asking for the data pointer if size is zero.
    if (s.size()) {
        auto data = static_cast<size_type *>(PyArray_DATA((PyArrayObject *)(ret)));
        for (decltype(s.size()) i = 0u; i < s.size(); ++i) {
            *(data + i + i) = s[i].first;
            *(data + i + i + 1u) = s[i].second;
        }
    }
    // Hand over to boost python.
    return bp::object(bp::handle<>(ret));
}

// Convert a numpy array of std::make_signed<vector_double::size_type>::type into a sparsity pattern.
inline pagmo::sparsity_pattern a_to_sp(PyArrayObject *o)
{
    using size_type = pagmo::vector_double::size_type;
    using int_type = std::make_signed<size_type>::type;
    if (!PyArray_ISCARRAY_RO(o)) {
        pygmo_throw(PyExc_RuntimeError,"cannot convert NumPy array to a sparsity pattern: "
         "data must be C-style contiguous, aligned, and in machine byte-order");
    }
    if (PyArray_NDIM(o) != 2) {
        pygmo_throw(PyExc_ValueError,"cannot convert NumPy array to a sparsity pattern: "
         "the array must be bidimensional");
    }
    if (PyArray_SHAPE(o)[1] != 2) {
        pygmo_throw(PyExc_ValueError,("cannot convert NumPy array to a sparsity pattern: "
         "the second dimension must be 2, but it is instead " + std::to_string(PyArray_SHAPE(o)[1])).c_str());
    }
    if (PyArray_TYPE(o) != cpp_npy<int_type>::value) {
        pygmo_throw(PyExc_TypeError,"cannot convert NumPy array to a sparsity pattern: "
         "the scalar type must be the signed counterpart of 'pagmo::vector_double::size_type'");
    }
    if (PyArray_STRIDES(o)[0] != sizeof(int_type) * 2u || PyArray_STRIDES(o)[1] != sizeof(int_type)) {
        pygmo_throw(PyExc_RuntimeError,"cannot convert NumPy array to a sparsity pattern: "
         "invalid strides detected");
    }
    if (PyArray_ITEMSIZE(o) != sizeof(int_type)) {
        pygmo_throw(PyExc_RuntimeError,("cannot convert NumPy array to a sparsity pattern: "
         "the size of the scalar type must be " + std::to_string(sizeof(int_type))).c_str());
    }
    const auto size = boost::numeric_cast<pagmo::sparsity_pattern::size_type>(PyArray_SHAPE(o)[0]);
    // Error handler for nice Python error messages.
    auto err_handler = [](const auto &n) {
        pygmo_throw(PyExc_OverflowError,("could not convert the sparsity index " + std::to_string(n) + " to the "
            "appropriate unsigned integer type").c_str());
    };
    if (size) {
        auto data = static_cast<int_type *>(PyArray_DATA(o));
        pagmo::sparsity_pattern retval;
        for (pagmo::sparsity_pattern::size_type i = 0u; i < size; ++i) {
            size_type a, b;
            try {
                a = boost::numeric_cast<size_type>(*(data + i + i));
            } catch (const std::bad_cast &) {
                err_handler(*(data + i + i));
            }
            try {
                b = boost::numeric_cast<size_type>(*(data + i + i + 1u));
            } catch (const std::bad_cast &) {
                err_handler(*(data + i + i + 1u));
            }
            retval.emplace_back(a,b);
        }
        return retval;
    }
    return pagmo::sparsity_pattern{};
}

// Try converting a python object to a sparsity pattern.
inline pagmo::sparsity_pattern to_sp(bp::object o)
{
    using size_type = pagmo::vector_double::size_type;
    bp::object l = builtin().attr("list");
    bp::object a = bp::import("numpy").attr("ndarray");
    if (isinstance(o,l)) {
        // Case 0: input object is a list.
        pagmo::sparsity_pattern retval;
        bp::stl_input_iterator<bp::tuple> begin(o), end;
        // Error handler to make better error messages in Python.
        auto err_handler = [](const auto &obj) {
            pygmo_throw(PyExc_RuntimeError,("couldn't extract a suitable sparsity index value from the object '" +
                str(obj) + "' of type '" + str(type(obj)) + "'.").c_str());
        };
        // Iterate over the list, trying to extract first a generic tuple from each element and then a pair
        // of appropriate integral values from each tuple's elements.
        for (; begin != end; ++begin) {
            bp::tuple tup;
            try {
                tup = *begin;
            } catch (...) {
                pygmo_throw(PyExc_TypeError,"a sparsity pattern represented as a list must be a list of tuples, "
                    "but a non-tuple element was encountered");
            }
            if (len(tup) != 2) {
                pygmo_throw(PyExc_ValueError,("invalid tuple size detected in sparsity pattern: it should be 2, "
                    "but it is " + std::to_string(len(tup)) + " instead").c_str());
            }
            size_type i, j;
            try {
                i = bp::extract<size_type>((tup)[0]);
            } catch (...) {
                err_handler((tup)[0]);
            }
            try {
                j = bp::extract<size_type>((tup)[1]);
            } catch (...) {
                err_handler((tup)[1]);
            }
            retval.emplace_back(i,j);
        }
        return retval;
    } else if (isinstance(o,a)) {
        // Case 1: input object is a NumPy array of some kind.
        // NOTE: the idea here is the following: we try to build a NumPy array of the signed counterpart of vector_double::size_type
        // (most likely long or long long) from whatever type of NumPy array was passed as input, and then we will convert
        // the elements to the appropriate size_type inside the a_to_sp routine. The reason for doing this is that
        // in typical usage Python integers are converted so signed integers when used inside NumPy arrays, so we want
        // to work with signed ints here as well in order no to force the user to create sparsity patterns
        // like array(...,dtype='ulonglong').
        auto n = PyArray_FROM_OTF(o.ptr(),cpp_npy<std::make_signed<size_type>::type>::value,NPY_ARRAY_IN_ARRAY);
        if (!n) {
            // NOTE: PyArray_FROM_OTF already sets the exception at the Python level with an appropriate message,
            // so we just throw the Python exception.
            bp::throw_error_already_set();
        }
        // Hand over to BP for nice RAII and exception safety.
        auto bp_n = bp::object(bp::handle<>(n));
        return a_to_sp((PyArrayObject *)bp_n.ptr());
    }
    pygmo_throw(PyExc_TypeError,("cannot convert the type '" + str(type(o)) + "' to a "
        "sparsity pattern: only lists of pairs of ints and NumPy arrays of ints "
        "are supported").c_str());
}

}

#endif
