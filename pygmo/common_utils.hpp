/* Copyright 2017 PaGMO development team

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

#ifndef PYGMO_COMMON_UTILS_HPP
#define PYGMO_COMMON_UTILS_HPP

#include "python_includes.hpp"

#include <algorithm>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/handle.hpp>
#include <boost/python/import.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/tuple.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

#include "numpy.hpp"

// A throwing macro similar to pagmo_throw, only for Python. This will set the global
// error string of Python to "msg", the exception type to "type", and then invoke the Boost
// Python function to raise the Python exception.
#define pygmo_throw(type, msg)                                                                                         \
    PyErr_SetString(type, msg);                                                                                        \
    boost::python::throw_error_already_set();                                                                          \
    throw

namespace pygmo
{

namespace bp = boost::python;

// Map C++ types to NPY_ types.
template <typename T>
struct cpp_npy {
};

#define PYGMO_CPP_NPY(from, to)                                                                                        \
    template <>                                                                                                        \
    struct cpp_npy<from> {                                                                                             \
        static constexpr auto value = to;                                                                              \
    };

// We only need the types below at the moment.
PYGMO_CPP_NPY(unsigned char, NPY_UBYTE)
PYGMO_CPP_NPY(unsigned short, NPY_USHORT)
PYGMO_CPP_NPY(unsigned, NPY_UINT)
PYGMO_CPP_NPY(unsigned long, NPY_ULONG)
PYGMO_CPP_NPY(unsigned long long, NPY_ULONGLONG)
PYGMO_CPP_NPY(signed char, NPY_BYTE)
PYGMO_CPP_NPY(short, NPY_SHORT)
PYGMO_CPP_NPY(int, NPY_INT)
PYGMO_CPP_NPY(long, NPY_LONG)
PYGMO_CPP_NPY(long long, NPY_LONGLONG)
PYGMO_CPP_NPY(float, NPY_FLOAT)
PYGMO_CPP_NPY(double, NPY_DOUBLE)

#undef PYGMO_CPP_NPY

// Perform a deep copy of input object o.
inline bp::object deepcopy(const bp::object &o)
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

// hasattr() wrapper.
inline bool hasattr(const bp::object &o, const char *name)
{
    return bp::extract<bool>(builtin().attr("hasattr")(o, name));
}

// Get the type of an object.
inline bp::object type(const bp::object &o)
{
    return builtin().attr("type")(o);
}

// String representation of an object.
inline std::string str(const bp::object &o)
{
    return bp::extract<std::string>(builtin().attr("str")(o));
}

// Check if type is callable.
inline bool callable(const bp::object &o)
{
    if (!o) {
        return false;
    }
    return bp::extract<bool>(builtin().attr("callable")(o));
}

// Convert a vector of arithmetic types into a 1D numpy array.
template <typename T>
using v_to_a_enabler = pagmo::enable_if_t<std::is_arithmetic<T>::value, int>;

template <typename T, v_to_a_enabler<T> = 0>
inline bp::object v_to_a(const std::vector<T> &v)
{
    // The dimensions of the array to be created.
    npy_intp dims[] = {boost::numeric_cast<npy_intp>(v.size())};
    // Attempt creating the array.
    PyObject *ret = PyArray_SimpleNew(1, dims, cpp_npy<T>::value);
    if (!ret) {
        pygmo_throw(PyExc_RuntimeError, "couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    // Hand over to BP for exception-safe behaviour.
    bp::object retval{bp::handle<>(ret)};
    if (v.size()) {
        // Copy over the data.
        std::copy(v.begin(), v.end(), static_cast<T *>(PyArray_DATA((PyArrayObject *)(ret))));
    }
    // Hand over to boost python.
    return retval;
}

// Convert a vector of vectors of arithmetic types into a 2D numpy array.
template <typename T>
using vv_to_a_enabler = pagmo::enable_if_t<std::is_arithmetic<T>::value, int>;

template <typename T, vv_to_a_enabler<T> = 0>
inline bp::object vv_to_a(const std::vector<std::vector<T>> &v)
{
    // The dimensions of the array to be created.
    const auto nrows = v.size();
    const auto ncols = nrows ? v[0].size() : 0u;
    npy_intp dims[] = {boost::numeric_cast<npy_intp>(nrows), boost::numeric_cast<npy_intp>(ncols)};
    // Attempt creating the array.
    PyObject *ret = PyArray_SimpleNew(2, dims, cpp_npy<T>::value);
    if (!ret) {
        pygmo_throw(PyExc_RuntimeError, "couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    // Hand over to BP for exception-safe behaviour.
    bp::object retval{bp::handle<>(ret)};
    if (nrows) {
        auto data = static_cast<T *>(PyArray_DATA((PyArrayObject *)(ret)));
        for (const auto &i : v) {
            if (i.size() != ncols) {
                pygmo_throw(PyExc_ValueError, "cannot convert a vector of vectors to a NumPy 2D array "
                                              "if the vector instances don't have all the same size");
            }
            std::copy(i.begin(), i.end(), data);
            data += ncols;
        }
    }
    return retval;
}

// isinstance wrapper.
inline bool isinstance(const bp::object &o, const bp::object &t)
{
    return bp::extract<bool>(builtin().attr("isinstance")(o, t));
}

// Convert a numpy array of double to a vector_double.
inline pagmo::vector_double ad_to_vd(PyArrayObject *o)
{
    assert(PyArray_TYPE(o) == NPY_DOUBLE);
    using size_type = pagmo::vector_double::size_type;
    if (!PyArray_ISCARRAY_RO(o)) {
        pygmo_throw(PyExc_RuntimeError, "cannot convert NumPy array to a vector of doubles: "
                                        "data must be C-style contiguous, aligned, and in machine byte-order");
    }
    if (PyArray_NDIM(o) != 1) {
        pygmo_throw(PyExc_ValueError, "cannot convert NumPy array to a vector of doubles: "
                                      "the array must be unidimensional");
    }
    if (PyArray_STRIDES(o)[0] != sizeof(double)) {
        pygmo_throw(PyExc_RuntimeError, ("cannot convert NumPy array to a vector of doubles: "
                                         "the stride value must be "
                                         + std::to_string(sizeof(double)))
                                            .c_str());
    }
    if (PyArray_ITEMSIZE(o) != sizeof(double)) {
        pygmo_throw(PyExc_RuntimeError, ("cannot convert NumPy array to a vector of doubles: "
                                         "the size of the scalar type must be "
                                         + std::to_string(sizeof(double)))
                                            .c_str());
    }
    // NOTE: not sure if this special casing is needed. We make sure
    // the array contains something in order to avoid messing around
    // with a potentially null pointer in the array.
    const auto size = boost::numeric_cast<size_type>(PyArray_SHAPE(o)[0]);
    if (size) {
        auto data = static_cast<double *>(PyArray_DATA(o));
        return pagmo::vector_double(data, data + size);
    }
    return pagmo::vector_double{};
}

// Convert an arbitrary python object to a vector_double.
inline pagmo::vector_double to_vd(const bp::object &o)
{
    bp::object a = bp::import("numpy").attr("ndarray");
    if (isinstance(o, a)) {
        // NOTE: the idea here is that we want to be able to convert
        // from a NumPy array of types other than double. This is useful
        // because one can then create arrays of ints and have them converted
        // on the fly (e.g., for the bounds). If the array is already a
        // double-precision array, this function should not do any copy.
        auto n = PyArray_FROM_OTF(o.ptr(), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!n) {
            bp::throw_error_already_set();
        }
        return ad_to_vd((PyArrayObject *)(bp::object(bp::handle<>(n)).ptr()));
    }
    // If o is not a numpy array, just try to iterate over it and extract doubles.
    bp::stl_input_iterator<double> begin(o), end;
    return pagmo::vector_double(begin, end);
}

// Convert a numpy array to a vector of vector_double.
inline std::vector<pagmo::vector_double> a_to_vvd(PyArrayObject *o)
{
    using size_type = std::vector<pagmo::vector_double>::size_type;
    if (!PyArray_ISCARRAY_RO(o)) {
        pygmo_throw(PyExc_RuntimeError, "cannot convert NumPy array to a vector of vector_double: "
                                        "data must be C-style contiguous, aligned, and in machine byte-order");
    }
    if (PyArray_NDIM(o) != 2) {
        pygmo_throw(PyExc_ValueError, "cannot convert NumPy array to a vector of vector_double: "
                                      "the array must be 2-dimensional");
    }
    if (PyArray_TYPE(o) != NPY_DOUBLE) {
        pygmo_throw(PyExc_TypeError, "cannot convert NumPy array to a vector of vector_double: "
                                     "the scalar type must be 'double'");
    }
    if (PyArray_ITEMSIZE(o) != sizeof(double)) {
        pygmo_throw(PyExc_RuntimeError, ("cannot convert NumPy array to a vector of vector_double: "
                                         "the size of the scalar type must be "
                                         + std::to_string(sizeof(double)))
                                            .c_str());
    }
    const auto size = boost::numeric_cast<size_type>(PyArray_SHAPE(o)[0]);
    std::vector<pagmo::vector_double> retval;
    if (size) {
        auto data = static_cast<double *>(PyArray_DATA(o));
        const auto ssize = PyArray_SHAPE(o)[1];
        for (size_type i = 0u; i < size; ++i, data += ssize) {
            retval.push_back(pagmo::vector_double(data, data + ssize));
        }
    }
    return retval;
}

// Convert an arbitrary Python object to a vector of vector_double.
inline std::vector<pagmo::vector_double> to_vvd(const bp::object &o)
{
    bp::object l = builtin().attr("list");
    bp::object a = bp::import("numpy").attr("ndarray");
    if (isinstance(o, l)) {
        bp::stl_input_iterator<bp::object> begin(o), end;
        std::vector<pagmo::vector_double> retval;
        for (; begin != end; ++begin) {
            retval.push_back(to_vd(*begin));
        }
        return retval;
    } else if (isinstance(o, a)) {
        auto n = PyArray_FROM_OTF(o.ptr(), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!n) {
            bp::throw_error_already_set();
        }
        return a_to_vvd((PyArrayObject *)(bp::object(bp::handle<>(n)).ptr()));
    }
    pygmo_throw(PyExc_TypeError, ("cannot convert the type '" + str(type(o))
                                  + "' to a "
                                    "vector of vector_double: only lists of doubles and NumPy arrays of doubles "
                                    "are supported")
                                     .c_str());
}

// Convert a numpy array to an std::vector<unsigned>.
inline std::vector<unsigned> a_to_vu(PyArrayObject *o)
{
    using size_type = std::vector<unsigned>::size_type;
    using int_type = std::make_signed<std::size_t>::type;
    if (!PyArray_ISCARRAY_RO(o)) {
        pygmo_throw(PyExc_RuntimeError, "cannot convert NumPy array to a vector of unsigned: "
                                        "data must be C-style contiguous, aligned, and in machine byte-order");
    }
    if (PyArray_NDIM(o) != 1) {
        pygmo_throw(PyExc_ValueError, "cannot convert NumPy array to a vector of unsigned: "
                                      "the array must be unidimensional");
    }
    if (PyArray_TYPE(o) != cpp_npy<int_type>::value) {
        pygmo_throw(PyExc_TypeError, "cannot convert NumPy array to a vector of unsigned: "
                                     "invalid scalar type");
    }
    if (PyArray_STRIDES(o)[0] != sizeof(int_type)) {
        pygmo_throw(PyExc_RuntimeError, ("cannot convert NumPy array to a vector of unsigned: "
                                         "the stride value must be "
                                         + std::to_string(sizeof(int_type)))
                                            .c_str());
    }
    if (PyArray_ITEMSIZE(o) != sizeof(int_type)) {
        pygmo_throw(PyExc_RuntimeError, ("cannot convert NumPy array to a vector of unsigned: "
                                         "the size of the scalar type must be "
                                         + std::to_string(sizeof(int_type)))
                                            .c_str());
    }
    const auto size = boost::numeric_cast<size_type>(PyArray_SHAPE(o)[0]);
    std::vector<unsigned> retval;
    if (size) {
        auto data = static_cast<int_type *>(PyArray_DATA(o));
        std::transform(data, data + size, std::back_inserter(retval),
                       [](int_type n) { return boost::numeric_cast<unsigned>(n); });
    }
    return retval;
}

// Convert an arbitrary python object to a vector of unsigned.
inline std::vector<unsigned> to_vu(const bp::object &o)
{
    bp::object l = builtin().attr("list");
    bp::object a = bp::import("numpy").attr("ndarray");
    if (isinstance(o, l)) {
        bp::stl_input_iterator<unsigned> begin(o), end;
        return std::vector<unsigned>(begin, end);
    } else if (isinstance(o, a)) {
        // NOTE: as usual, we try first to create an array of signed ints,
        // and we convert to unsigned in a_to_vu().
        using int_type = std::make_signed<std::size_t>::type;
        auto n = PyArray_FROM_OTF(o.ptr(), cpp_npy<int_type>::value, NPY_ARRAY_IN_ARRAY);
        if (!n) {
            bp::throw_error_already_set();
        }
        return a_to_vu((PyArrayObject *)(bp::object(bp::handle<>(n)).ptr()));
    }
    pygmo_throw(PyExc_TypeError, ("cannot convert the type '" + str(type(o))
                                  + "' to a vector of ints: only lists of ints and NumPy arrays of ints are supported")
                                     .c_str());
}

// Convert a sparsity pattern into a numpy array.
inline bp::object sp_to_a(const pagmo::sparsity_pattern &s)
{
    // The unsigned integral type that is used in the sparsity pattern.
    using size_type = pagmo::vector_double::size_type;
    // Its signed counterpart.
    using int_type = std::make_signed<size_type>::type;
    npy_intp dims[] = {boost::numeric_cast<npy_intp>(s.size()), 2};
    PyObject *ret = PyArray_SimpleNew(2, dims, cpp_npy<int_type>::value);
    if (!ret) {
        pygmo_throw(PyExc_RuntimeError, "couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    auto err_handler = [](const decltype(s[0].first) &n) {
        pygmo_throw(PyExc_OverflowError, ("overflow in the conversion of the sparsity index " + std::to_string(n)
                                          + " to the appropriate signed integer type")
                                             .c_str());
    };
    // NOTE: same as above, avoid asking for the data pointer if size is zero.
    if (s.size()) {
        auto data = static_cast<int_type *>(PyArray_DATA((PyArrayObject *)(ret)));
        for (decltype(s.size()) i = 0u; i < s.size(); ++i) {
            try {
                *(data + i + i) = boost::numeric_cast<int_type>(s[i].first);
            } catch (const std::bad_cast &) {
                err_handler(s[i].first);
            }
            try {
                *(data + i + i + 1u) = boost::numeric_cast<int_type>(s[i].second);
            } catch (const std::bad_cast &) {
                err_handler(s[i].second);
            }
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
        pygmo_throw(PyExc_RuntimeError, "cannot convert NumPy array to a sparsity pattern: "
                                        "data must be C-style contiguous, aligned, and in machine byte-order");
    }
    if (PyArray_NDIM(o) != 2) {
        pygmo_throw(PyExc_ValueError, "cannot convert NumPy array to a sparsity pattern: "
                                      "the array must be bidimensional");
    }
    if (PyArray_SHAPE(o)[1] != 2) {
        pygmo_throw(PyExc_ValueError, ("cannot convert NumPy array to a sparsity pattern: "
                                       "the second dimension must be 2, but it is instead "
                                       + std::to_string(PyArray_SHAPE(o)[1]))
                                          .c_str());
    }
    if (PyArray_TYPE(o) != cpp_npy<int_type>::value) {
        pygmo_throw(PyExc_TypeError,
                    "cannot convert NumPy array to a sparsity pattern: "
                    "the scalar type must be the signed counterpart of 'pagmo::vector_double::size_type'");
    }
    if (PyArray_STRIDES(o)[0] != sizeof(int_type) * 2u || PyArray_STRIDES(o)[1] != sizeof(int_type)) {
        pygmo_throw(PyExc_RuntimeError, "cannot convert NumPy array to a sparsity pattern: "
                                        "invalid strides detected");
    }
    if (PyArray_ITEMSIZE(o) != sizeof(int_type)) {
        pygmo_throw(PyExc_RuntimeError, ("cannot convert NumPy array to a sparsity pattern: "
                                         "the size of the scalar type must be "
                                         + std::to_string(sizeof(int_type)))
                                            .c_str());
    }
    const auto size = boost::numeric_cast<pagmo::sparsity_pattern::size_type>(PyArray_SHAPE(o)[0]);
    // Error handler for nice Python error messages.
    auto err_handler = [](int_type n) {
        pygmo_throw(PyExc_OverflowError, ("overflow in the conversion of the sparsity index " + std::to_string(n)
                                          + " to the "
                                            "appropriate unsigned integer type")
                                             .c_str());
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
            retval.emplace_back(a, b);
        }
        return retval;
    }
    return pagmo::sparsity_pattern{};
}

// Try converting a python object to a sparsity pattern.
inline pagmo::sparsity_pattern to_sp(const bp::object &o)
{
    using size_type = pagmo::vector_double::size_type;
    bp::object l = builtin().attr("list");
    bp::object a = bp::import("numpy").attr("ndarray");
    if (isinstance(o, l)) {
        // Case 0: input object is a list.
        pagmo::sparsity_pattern retval;
        bp::stl_input_iterator<bp::tuple> begin(o), end;
        // Error handler to make better error messages in Python.
        auto err_handler = [](const bp::object &obj) {
            pygmo_throw(PyExc_RuntimeError, ("couldn't extract a suitable sparsity index value from the object '"
                                             + str(obj) + "' of type '" + str(type(obj)) + "'.")
                                                .c_str());
        };
        // Iterate over the list, trying to extract first a generic tuple from each element and then a pair
        // of appropriate integral values from each tuple's elements.
        bp::tuple tup;
        for (; begin != end; ++begin) {
            try {
                tup = *begin;
            } catch (...) {
                pygmo_throw(PyExc_TypeError, "a sparsity pattern represented as a list must be a list of tuples, "
                                             "but a non-tuple element was encountered");
            }
            if (len(tup) != 2) {
                pygmo_throw(PyExc_ValueError, ("invalid tuple size detected in sparsity pattern: it should be 2, "
                                               "but it is "
                                               + std::to_string(len(tup)) + " instead")
                                                  .c_str());
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
            retval.emplace_back(i, j);
        }
        return retval;
    } else if (isinstance(o, a)) {
        // Case 1: input object is a NumPy array of some kind.
        // NOTE: the idea here is the following: we try to build a NumPy array of the signed counterpart of
        // vector_double::size_type
        // (most likely long or long long) from whatever type of NumPy array was passed as input, and then we will
        // convert
        // the elements to the appropriate size_type inside the a_to_sp routine. The reason for doing this is that
        // in typical usage Python integers are converted so signed integers when used inside NumPy arrays, so we want
        // to work with signed ints here as well in order no to force the user to create sparsity patterns
        // like array(...,dtype='ulonglong').
        auto n = PyArray_FROM_OTF(o.ptr(), cpp_npy<std::make_signed<size_type>::type>::value, NPY_ARRAY_IN_ARRAY);
        if (!n) {
            // NOTE: PyArray_FROM_OTF already sets the exception at the Python level with an appropriate message,
            // so we just throw the Python exception.
            bp::throw_error_already_set();
        }
        // Hand over to BP for nice RAII and exception safety.
        auto bp_n = bp::object(bp::handle<>(n));
        return a_to_sp((PyArrayObject *)bp_n.ptr());
    }
    pygmo_throw(PyExc_TypeError, ("cannot convert the type '" + str(type(o))
                                  + "' to a "
                                    "sparsity pattern: only lists of pairs of ints and NumPy arrays of ints "
                                    "are supported")
                                     .c_str());
}

// Wrapper around the CPython function to create a bytes object from raw data.
bp::object make_bytes(const char *ptr, Py_ssize_t len)
{
    PyObject *retval;
    if (len) {
        retval = PyBytes_FromStringAndSize(ptr, len);
    } else {
        retval = PyBytes_FromStringAndSize(nullptr, 0);
    }
    if (!retval) {
        pygmo_throw(PyExc_RuntimeError, "unable to create a bytes object: the 'PyBytes_FromStringAndSize()' "
                                        "function returned NULL");
    }
    return bp::object(bp::handle<>(retval));
}

// Generic copy wrappers.
template <typename T>
inline T generic_copy_wrapper(const T &x)
{
    return x;
}

template <typename T>
inline T generic_deepcopy_wrapper(const T &x, bp::dict)
{
    return x;
}

// Generic extract() wrappers.
template <typename C, typename T>
inline T generic_cpp_extract(const C &c, const T &)
{
    auto ptr = c.template extract<T>();
    if (!ptr) {
        // TODO: demangler?
        pygmo_throw(PyExc_TypeError, "");
    }
    return *ptr;
}

template <typename C>
inline bp::object generic_py_extract(const C &c, const bp::object &t)
{
    auto ptr = c.template extract<bp::object>();
    if (!ptr) {
        pygmo_throw(PyExc_TypeError, "could not extract a Python object: "
                                     "the inner object is a C++ exposed type");
    }
    if (type(*ptr) != t) {
        pygmo_throw(PyExc_TypeError, ("the inner object is not of type " + str(t)).c_str());
    }
    return deepcopy(*ptr);
}

// Detail implementation of the tuple conversion below.
namespace detail
{

template <typename Func, typename Tup, std::size_t... index>
auto ct2pt_invoke_helper(Func &&func, Tup &&tup, pagmo::detail::index_sequence<index...>)
    -> decltype(func(std::get<index>(std::forward<Tup>(tup))...))
{
    return func(std::get<index>(std::forward<Tup>(tup))...);
}

template <typename Func, typename Tup>
auto ct2pt_invoke(Func &&func, Tup &&tup)
    -> decltype(ct2pt_invoke_helper(std::forward<Func>(func), std::forward<Tup>(tup),
                                    pagmo::detail::make_index_sequence<std::tuple_size<pagmo::decay_t<Tup>>::value>{}))
{
    return ct2pt_invoke_helper(std::forward<Func>(func), std::forward<Tup>(tup),
                               pagmo::detail::make_index_sequence<std::tuple_size<pagmo::decay_t<Tup>>::value>{});
}
}

// Utility function to convert a C++ tuple into a Python tuple.
template <typename... Args>
inline bp::tuple cpptuple_to_pytuple(const std::tuple<Args...> &t)
{
    return detail::ct2pt_invoke(bp::make_tuple<Args...>, t);
}
}

#endif
