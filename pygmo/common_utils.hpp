/* Copyright 2017-2018 PaGMO development team

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

#include <pygmo/python_includes.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/class.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/handle.hpp>
#include <boost/python/import.hpp>
#include <boost/python/list.hpp>
#include <boost/python/make_function.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/tuple.hpp>

#include <pagmo/exceptions.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

#include <pygmo/numpy.hpp>

#if defined(_MSC_VER)

#include <pygmo/function_traits.hpp>

#endif

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
    if (o.is_none()) {
        return false;
    }
    return bp::extract<bool>(builtin().attr("callable")(o));
}

// Check if 'o' has a callable attribute (i.e., a method) named 's'. If so, it will
// return the attribute, otherwise it will return None.
inline bp::object callable_attribute(const bp::object &o, const char *s)
{
    if (hasattr(o, s)) {
        bp::object retval = o.attr(s);
        if (callable(retval)) {
            return retval;
        }
    }
    return bp::object();
}

// Convert a vector of arithmetic types into a 1D numpy array.
// NOTE: provide two overloads, one for unsigned integral types and the
// other for the other arithmetic types. The idea is that we may not want
// to produce numpy arrays of unsigned integrals, as they are more painful
// to deal with in Python than just plain signed integrals.
template <typename T, pagmo::enable_if_t<pagmo::detail::disjunction<
                                             std::is_floating_point<T>,
                                             pagmo::detail::conjunction<std::is_integral<T>, std::is_signed<T>>>::value,
                                         int> = 0>
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
        std::copy(v.begin(), v.end(), static_cast<T *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ret))));
    }
    // Hand over to boost python.
    return retval;
}

template <typename T,
          pagmo::enable_if_t<pagmo::detail::conjunction<std::is_integral<T>, std::is_unsigned<T>>::value, int> = 0>
inline bp::object v_to_a(const std::vector<T> &v, bool keep_unsigned = false)
{
    // The dimensions of the array to be created.
    npy_intp dims[] = {boost::numeric_cast<npy_intp>(v.size())};
    // Attempt creating the array. The ndtype will be the signed counterpart of T,
    // if keep_unsigned is false.
    using int_type = typename std::make_signed<T>::type;
    PyObject *ret = PyArray_SimpleNew(1, dims, keep_unsigned ? cpp_npy<T>::value : cpp_npy<int_type>::value);
    if (!ret) {
        pygmo_throw(PyExc_RuntimeError, "couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    // Hand over to BP for exception-safe behaviour.
    bp::object retval{bp::handle<>(ret)};
    if (v.size()) {
        if (keep_unsigned) {
            // Copy over the data.
            std::copy(v.begin(), v.end(), static_cast<T *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ret))));
        } else {
            // Copy over the data, transforming from unsigned to signed on the fly.
            std::transform(v.begin(), v.end(),
                           static_cast<int_type *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ret))),
                           [](const T &n) { return boost::numeric_cast<int_type>(n); });
        }
    }
    // Hand over to boost python.
    return retval;
}

// Convert a vector of vectors of arithmetic types into a 2D numpy array.
// NOTE: same scheme as above.
template <typename T, pagmo::enable_if_t<pagmo::detail::disjunction<
                                             std::is_floating_point<T>,
                                             pagmo::detail::conjunction<std::is_integral<T>, std::is_signed<T>>>::value,
                                         int> = 0>
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
        auto data = static_cast<T *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ret)));
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

template <typename T,
          pagmo::enable_if_t<pagmo::detail::conjunction<std::is_integral<T>, std::is_unsigned<T>>::value, int> = 0>
inline bp::object vv_to_a(const std::vector<std::vector<T>> &v, bool keep_unsigned = false)
{
    // The dimensions of the array to be created.
    const auto nrows = v.size();
    const auto ncols = nrows ? v[0].size() : 0u;
    npy_intp dims[] = {boost::numeric_cast<npy_intp>(nrows), boost::numeric_cast<npy_intp>(ncols)};
    // Attempt creating the array. The ndtype will be the signed counterpart of T,
    // if keep_unsigned is false.
    using int_type = typename std::make_signed<T>::type;
    PyObject *ret = PyArray_SimpleNew(2, dims, keep_unsigned ? cpp_npy<T>::value : cpp_npy<int_type>::value);
    if (!ret) {
        pygmo_throw(PyExc_RuntimeError, "couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    // Hand over to BP for exception-safe behaviour.
    bp::object retval{bp::handle<>(ret)};
    if (nrows) {
        if (keep_unsigned) {
            auto data = static_cast<T *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ret)));
            for (const auto &i : v) {
                if (i.size() != ncols) {
                    pygmo_throw(PyExc_ValueError, "cannot convert a vector of vectors to a NumPy 2D array "
                                                  "if the vector instances don't have all the same size");
                }
                std::copy(i.begin(), i.end(), data);
                data += ncols;
            }
        } else {
            auto data = static_cast<int_type *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ret)));
            for (const auto &i : v) {
                if (i.size() != ncols) {
                    pygmo_throw(PyExc_ValueError, "cannot convert a vector of vectors to a NumPy 2D array "
                                                  "if the vector instances don't have all the same size");
                }
                std::transform(i.begin(), i.end(), data, [](const T &n) { return boost::numeric_cast<int_type>(n); });
                data += ncols;
            }
        }
        return retval;
    }
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
        pygmo_throw(PyExc_ValueError, ("cannot convert NumPy array to a vector of doubles: "
                                       "the array must be unidimensional, but the dimension is "
                                       + std::to_string(PyArray_NDIM(o)) + " instead")
                                          .c_str());
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
        return ad_to_vd(reinterpret_cast<PyArrayObject *>(bp::object(bp::handle<>(n)).ptr()));
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
        return a_to_vvd(reinterpret_cast<PyArrayObject *>(bp::object(bp::handle<>(n)).ptr()));
    }
    pygmo_throw(PyExc_TypeError, ("cannot convert the type '" + str(type(o))
                                  + "' to a vector of vector_double: only lists of doubles and NumPy arrays of doubles "
                                    "are supported")
                                     .c_str());
}

// Convert a 1D numpy array of input integral type SourceInt to an std::vector of
// unsigned integral type UInt.
template <typename UInt, typename SourceInt>
inline std::vector<UInt> a_to_vuint(PyArrayObject *o)
{
    static_assert(std::is_integral<UInt>::value && std::is_unsigned<UInt>::value,
                  "The UInt type must be an unsigned integral type.");

    static_assert(std::is_integral<SourceInt>::value, "The SourceInt type must be an integral type.");

    using size_type = typename std::vector<UInt>::size_type;

    if (!PyArray_ISCARRAY_RO(o)) {
        pygmo_throw(PyExc_RuntimeError, "cannot convert a NumPy array to a vector of unsigned integrals: "
                                        "data must be C-style contiguous, aligned, and in machine byte-order");
    }
    if (PyArray_NDIM(o) != 1) {
        pygmo_throw(PyExc_ValueError, "cannot convert a NumPy array to a vector of unsigned integrals: "
                                      "the array must be unidimensional");
    }
    if (PyArray_TYPE(o) != cpp_npy<SourceInt>::value) {
        pygmo_throw(PyExc_TypeError, "cannot convert a NumPy array to a vector of unsigned integrals: "
                                     "the input scalar type is inconsistent with the expected integral type");
    }
    if (PyArray_STRIDES(o)[0] != sizeof(SourceInt)) {
        pygmo_throw(PyExc_RuntimeError, ("cannot convert a NumPy array to a vector of unsigned integrals: "
                                         "the stride value must be "
                                         + std::to_string(sizeof(SourceInt)))
                                            .c_str());
    }
    if (PyArray_ITEMSIZE(o) != sizeof(SourceInt)) {
        pygmo_throw(PyExc_RuntimeError, ("cannot convert a NumPy array to a vector of unsigned integrals: "
                                         "the size of the scalar type must be "
                                         + std::to_string(sizeof(SourceInt)))
                                            .c_str());
    }

    const auto size = boost::numeric_cast<size_type>(PyArray_SHAPE(o)[0]);
    std::vector<UInt> retval;
    retval.resize(boost::numeric_cast<decltype(retval.size())>(size));

    if (size) {
        auto data = static_cast<SourceInt *>(PyArray_DATA(o));
        std::transform(data, data + size, retval.data(), [](SourceInt n) { return boost::numeric_cast<UInt>(n); });
    }

    return retval;
}

// Convert an arbitrary python object to a vector of unsigned integrals.
// This function expects in input either a list of something convertible
// to UInt, or a NumPy array of *signed* integrals. The point of expecting
// signed integrals is that, except in rare occasions, we just want to
// deal with signed integral types on the Python side and convert back
// to unsigned only when crossing over to C++. This is much more natural
// for the user, who does not have to deal with the creation of NumPy
// arrays with a non-default dtype, etc. The only exception to this
// rule is when we are dealing with individual IDs, which must be kept
// unsigned.
template <typename UInt>
inline std::vector<UInt> to_vuint(const bp::object &o)
{
    static_assert(std::is_integral<UInt>::value && std::is_unsigned<UInt>::value,
                  "This function must be used only with unsigned integral types.");

    bp::object l = builtin().attr("list");
    bp::object a = bp::import("numpy").attr("ndarray");

    if (isinstance(o, l)) {
        bp::stl_input_iterator<UInt> begin(o), end;
        return std::vector<UInt>(begin, end);
    } else if (isinstance(o, a)) {
        // NOTE: the idea here is that we expect in input a numpy array of
        // the "natural sized" signed int type for the platform, which we
        // heuristically assume to be the signed counterpart of size_t. The idea
        // is that we basically want the users to use signed integral arrays
        // when talking to pygmo, because they are the easiest to work with
        // and I don't see much benefit in making the distinction between signed/unsigned
        // in Python. We can rework this approach if it turns out to be too restrictive.
        using int_type = std::make_signed<std::size_t>::type;

        auto n = PyArray_FROM_OTF(o.ptr(), cpp_npy<int_type>::value, NPY_ARRAY_IN_ARRAY);
        if (!n) {
            bp::throw_error_already_set();
        }

        return a_to_vuint<UInt, int_type>(reinterpret_cast<PyArrayObject *>(bp::object(bp::handle<>(n)).ptr()));
    }
    pygmo_throw(PyExc_TypeError,
                ("cannot convert the type '" + str(type(o))
                 + "' to a vector of unsigned integrals: only lists of ints and NumPy arrays of ints are supported")
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
    // Hand over to BP for exception-safe behaviour.
    bp::object retval{bp::handle<>(ret)};
    auto err_handler = [](const decltype(s[0].first) &n) {
        pygmo_throw(PyExc_OverflowError, ("overflow in the conversion of the sparsity index " + std::to_string(n)
                                          + " to the appropriate signed integer type")
                                             .c_str());
    };
    // NOTE: same as above, avoid asking for the data pointer if size is zero.
    if (s.size()) {
        auto data = static_cast<int_type *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ret)));
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
    return retval;
}

// Convert a numpy array of std::make_signed<vector_double::size_type>::type into a sparsity pattern.
inline pagmo::sparsity_pattern a_to_sp(PyArrayObject *o)
{
    using size_type = pagmo::vector_double::size_type;
    using int_type = std::make_signed<size_type>::type;
    if (!PyArray_ISCARRAY_RO(o)) {
        pygmo_throw(PyExc_ValueError, "cannot convert NumPy array to a sparsity pattern: "
                                      "data must be C-style contiguous, aligned, and in machine byte-order");
    }
    if (PyArray_NDIM(o) != 2) {
        pygmo_throw(PyExc_ValueError, ("cannot convert NumPy array to a sparsity pattern: "
                                       "the array must be bidimensional, but its dimension is "
                                       + std::to_string(PyArray_NDIM(o)) + " instead")
                                          .c_str());
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
        pygmo_throw(PyExc_ValueError, "cannot convert NumPy array to a sparsity pattern: "
                                      "invalid strides detected");
    }
    if (PyArray_ITEMSIZE(o) != sizeof(int_type)) {
        pygmo_throw(PyExc_ValueError, ("cannot convert NumPy array to a sparsity pattern: "
                                       "the size of the scalar type must be "
                                       + std::to_string(sizeof(int_type)))
                                          .c_str());
    }
    const auto size = boost::numeric_cast<pagmo::sparsity_pattern::size_type>(PyArray_SHAPE(o)[0]);
    // Error handler for nice Python error messages.
    auto err_handler = [](int_type n) {
        pygmo_throw(PyExc_OverflowError, ("overflow in the conversion of the sparsity index " + std::to_string(n)
                                          + " to the appropriate unsigned integer type")
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
    if (isinstance(o, bp::import("numpy").attr("ndarray"))) {
        // Input object is a NumPy array of some kind.
        // NOTE: the idea here is the following: we try to build a NumPy array of the signed counterpart of
        // vector_double::size_type (most likely long or long long) from whatever type of NumPy array was passed as
        // input, and then we will convert the elements to the appropriate size_type inside the a_to_sp routine. The
        // reason for doing this is that in typical usage Python integers are converted to signed integers when used
        // inside NumPy arrays, so we want to work with signed ints here as well in order no to force the user to
        // create sparsity patterns like array(...,dtype='ulonglong').
        auto n = PyArray_FROM_OTF(o.ptr(), cpp_npy<std::make_signed<size_type>::type>::value, NPY_ARRAY_IN_ARRAY);
        if (!n) {
            // NOTE: PyArray_FROM_OTF already sets the exception at the Python level with an appropriate message,
            // so we just throw the Python exception.
            bp::throw_error_already_set();
        }
        // Hand over to BP for nice RAII and exception safety.
        auto bp_n = bp::object(bp::handle<>(n));
        return a_to_sp(reinterpret_cast<PyArrayObject *>(bp_n.ptr()));
    }
    pagmo::sparsity_pattern retval;
    // We will try to interpret o as a collection of generic python objects, and each element
    // of o as another collection of python objects.
    bp::stl_input_iterator<bp::object> begin(o), end;
    std::array<size_type, 2> tmp_arr;
    for (; begin != end; ++begin) {
        // Inside each element of the collection, we try to iterate over 2 elements.
        bp::stl_input_iterator<bp::object> begin2(*begin), end2;
        std::size_t i = 0;
        for (; begin2 != end2; ++begin2, ++i) {
            if (i == 2u) {
                // This means that the element of the sparsity pattern is not a pair (i,j) of 2 values,
                // it contains > 2 values.
                pygmo_throw(PyExc_ValueError,
                            ("in the construction of a sparsity pattern, the sparsity pattern element '" + str(*begin)
                             + "' of type '" + str(type(*begin))
                             + "' was detected to contain more than 2 values, but elements of "
                               "sparsity patterns need to consist exactly of 2 values")
                                .c_str());
            }
            tmp_arr[i] = bp::extract<size_type>(*begin2);
        }
        if (i < 2u) {
            // This means that the sparsity pattern element containes 0 or 1 values, whereas
            // it needs to contain exactly 2 values.
            pygmo_throw(PyExc_ValueError,
                        ("in the construction of a sparsity pattern, the sparsity pattern element '" + str(*begin)
                         + "' of type '" + str(type(*begin)) + "' was detected to contain " + std::to_string(i)
                         + " values, but elements of sparsity patterns need to consist exactly of 2 values")
                            .c_str());
        }
        // Add the sparsity pattern element to the retval.
        retval.emplace_back(tmp_arr[0], tmp_arr[1]);
    }
    return retval;
}

// Wrapper around the CPython function to create a bytes object from raw data.
inline bp::object make_bytes(const char *ptr, Py_ssize_t len)
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
inline T *generic_cpp_extract(C &c, const T &)
{
    return c.template extract<T>();
}

template <typename C>
inline bp::object generic_py_extract(C &c, const bp::object &t)
{
    auto ptr = c.template extract<bp::object>();
    if (ptr && (t == type(*ptr) || t == builtin().attr("object"))) {
        // c contains a user-defined pythonic entity and either:
        // - the type passed in by the user is the exact type of the user-defined
        //   entity, or
        // - the user supplied as t the builtin 'object' type (which we use as a
        //   wildcard for any Python type).
        // Let's return the extracted object.
        return *ptr;
    }
    // Either the user-defined entity is not pythonic, or the user specified the
    // wrong type. Return None.
    return bp::object{};
}

// Detail implementation of the tuple conversion below.
namespace detail
{

template <typename Func, typename Tup, std::size_t... index>
inline auto ct2pt_invoke_helper(Func &&func, Tup &&tup, pagmo::detail::index_sequence<index...>)
    -> decltype(func(std::get<index>(std::forward<Tup>(tup))...))
{
    return func(std::get<index>(std::forward<Tup>(tup))...);
}

template <typename Func, typename Tup>
inline auto ct2pt_invoke(Func &&func, Tup &&tup)
    -> decltype(ct2pt_invoke_helper(std::forward<Func>(func), std::forward<Tup>(tup),
                                    pagmo::detail::make_index_sequence<std::tuple_size<pagmo::decay_t<Tup>>::value>{}))
{
    return ct2pt_invoke_helper(std::forward<Func>(func), std::forward<Tup>(tup),
                               pagmo::detail::make_index_sequence<std::tuple_size<pagmo::decay_t<Tup>>::value>{});
}
} // namespace detail

// Utility function to convert a C++ tuple into a Python tuple.
template <typename... Args>
inline bp::tuple cpptuple_to_pytuple(const std::tuple<Args...> &t)
{
    return detail::ct2pt_invoke(bp::make_tuple<Args...>, t);
}

// This RAII struct will unlock the GIL on construction,
// and lock it again on destruction.
//
// See: https://docs.python.org/2/c-api/init.html
struct gil_releaser {
    gil_releaser()
    {
        m_thread_state = ::PyEval_SaveThread();
    }
    // Make sure we don't accidentally try to copy/move it.
    gil_releaser(const gil_releaser &) = delete;
    gil_releaser(gil_releaser &&) = delete;
    gil_releaser &operator=(const gil_releaser &) = delete;
    gil_releaser &operator=(gil_releaser &&) = delete;
    ~gil_releaser()
    {
        ::PyEval_RestoreThread(m_thread_state);
    }
    ::PyThreadState *m_thread_state;
};

// This RAII struct ensures that the Python interpreter
// can be used from a thread created from outside Python
// (e.g., a pthread/std::thread/etc. created from C/C++).
// On creation, it will register the C/C++ thread with
// the Python interpreter and lock the GIL. On destruction,
// it will release any resource acquired on construction
// and unlock the GIL.
//
// See: https://docs.python.org/2/c-api/init.html
struct gil_thread_ensurer {
    gil_thread_ensurer()
    {
        m_state = ::PyGILState_Ensure();
    }
    // Make sure we don't accidentally try to copy/move it.
    gil_thread_ensurer(const gil_thread_ensurer &) = delete;
    gil_thread_ensurer(gil_thread_ensurer &&) = delete;
    gil_thread_ensurer &operator=(const gil_thread_ensurer &) = delete;
    gil_thread_ensurer &operator=(gil_thread_ensurer &&) = delete;
    ~gil_thread_ensurer()
    {
        ::PyGILState_Release(m_state);
    }
    ::PyGILState_STATE m_state;
};

#if defined(_MSC_VER)

template <typename T, std::size_t... I>
auto lcast_impl(T func, std::index_sequence<I...>)
{
    using ftraits = utils::function_traits<T>;
    using ret_t = typename ftraits::result_type;
    return static_cast<ret_t (*)(typename ftraits::template arg<I>::type...)>(func);
}

template <typename T>
inline auto lcast(T func)
{
    using ftraits = utils::function_traits<T>;
    constexpr std::size_t arity = ftraits::arity;
    using seq_t = std::make_index_sequence<arity>;
    return lcast_impl(func, seq_t{});
}

#else

template <typename T>
inline auto lcast(T func) -> decltype(+func)
{
    return +func;
}

#endif

// NOTE: these are alternative implementations of BP's add_property() functionality for classes.
// The reason they exist (and why they should be used instead of the BP implementation) is because
// we are running into a nasty crash on MinGW upon module import that I did not manage to debug fully, but which
// seems to be somehow related to BP's add_property() (at least judging from the limited stacktrace I could obtain
// on windows). These alternative wrappers seem to sidestep the issue, at least so far. They can be used exactly
// like BP's add_property(), the only difference being that they are functions rather than methods, and they thus
// require the BP class to be passed in as first argument.
template <typename T>
inline void add_property(bp::class_<T> &c, const char *name, const bp::object &getter, const char *doc = "")
{
    c.setattr(name, builtin().attr("property")(getter, bp::object(), bp::object(), doc));
}

template <typename T, typename G>
inline void add_property(bp::class_<T> &c, const char *name, G getter, const char *doc = "")
{
    add_property(c, name, bp::make_function(getter), doc);
}

template <typename T>
inline void add_property(bp::class_<T> &c, const char *name, const bp::object &getter, const bp::object &setter,
                         const char *doc = "")
{
    c.setattr(name, builtin().attr("property")(getter, setter, bp::object(), doc));
}

template <typename T, typename G, typename S>
inline void add_property(bp::class_<T> &c, const char *name, G getter, S setter, const char *doc = "")
{
    add_property(c, name, bp::make_function(getter), bp::make_function(setter), doc);
}

template <typename T, typename S>
inline void add_property(bp::class_<T> &c, const char *name, const bp::object &getter, S setter, const char *doc = "")
{
    add_property(c, name, getter, bp::make_function(setter), doc);
}

template <typename T, typename G>
inline void add_property(bp::class_<T> &c, const char *name, G getter, const bp::object &setter, const char *doc = "")
{
    add_property(c, name, bp::make_function(getter), setter, doc);
}

// A small helper to get the list of currently-registered APs.
inline bp::list get_ap_list()
{
    // Get the list of registered APs.
    auto &ap_set = *reinterpret_cast<std::unordered_set<std::string> *>(
        bp::extract<std::uintptr_t>(bp::import("pygmo").attr("core").attr("_ap_set_address"))());
    bp::list retval;
    for (const auto &s : ap_set) {
        retval.append(s);
    }
    return retval;
}

// Try to  import all the APs listed in l. This is used when deserializing a pygmo class.
// If an AP cannot be imported, ignore the error and move to the next AP.
inline void import_aps(const bp::list &l)
{
    bp::stl_input_iterator<std::string> begin(l), end;
    for (; begin != end; ++begin) {
        try {
            bp::import(begin->c_str());
        } catch (const bp::error_already_set &) {
            assert(::PyErr_Occurred());
            if (::PyErr_ExceptionMatches(::PyExc_ImportError)) {
                ::PyErr_Clear();
            } else {
                throw;
            }
        }
    }
}

// Convert an individuals_group_t into a Python tuple of:
// - 1D integral array of IDs,
// - 2D float array of dvs,
// - 2D float array of fvs.
inline bp::tuple inds_to_tuple(const pagmo::individuals_group_t &inds)
{
    // Do the IDs.
    // NOTE: as usual, keep the IDs as unsigned values.
    auto ID_arr = v_to_a(std::get<0>(inds), true);

    // Decision vectors.
    auto dv_arr = vv_to_a(std::get<1>(inds));

    // Fitness vectors.
    auto fv_arr = vv_to_a(std::get<2>(inds));

    return bp::make_tuple(ID_arr, dv_arr, fv_arr);
}

// Convert a generic Python object into a vector of individual IDs.
// NOTE: this is very similar to to_vuint(), but with the difference
// that we want to keep the unsignedness of the IDs. Perhaps
// in the future we can refactor these functions to have less code
// duplication.
inline std::vector<unsigned long long> to_ID_vector(const bp::object &o)
{
    bp::object l = builtin().attr("list");
    bp::object a = bp::import("numpy").attr("ndarray");

    if (isinstance(o, l)) {
        bp::stl_input_iterator<unsigned long long> begin(o), end;
        return std::vector<unsigned long long>(begin, end);
    } else if (isinstance(o, a)) {
        auto n = PyArray_FROM_OTF(o.ptr(), cpp_npy<unsigned long long>::value, NPY_ARRAY_IN_ARRAY);
        if (!n) {
            bp::throw_error_already_set();
        }

        return a_to_vuint<unsigned long long, unsigned long long>(
            reinterpret_cast<PyArrayObject *>(bp::object(bp::handle<>(n)).ptr()));
    }
    pygmo_throw(PyExc_TypeError, ("cannot convert the type '" + str(type(o))
                                  + "' to a vector of individual IDs: only lists of unsigned ints and NumPy arrays of "
                                    "unsigned ints are supported")
                                     .c_str());
}

// Convert a generic Python object into an individuals_group_t.
inline pagmo::individuals_group_t to_inds(const bp::object &o)
{
    // We assume we receive in input something we can iterate over.
    bp::stl_input_iterator<bp::object> begin(o), end;

    if (begin == end) {
        // Empty iteratable.
        pygmo_throw(PyExc_ValueError, "cannot convert an empty iteratable into a pagmo::individuals_group_t");
    }

    // Try fetching the IDs.
    auto ID_vec = to_ID_vector(*begin);

    if (++begin == end) {
        // Iteratable with only 1 element.
        pygmo_throw(PyExc_ValueError, "cannot convert an iteratable with only 1 element into a "
                                      "pagmo::individuals_group_t (exactly 3 elements are needed)");
    }

    // Try fetching the decision vectors.
    auto dvs_vec = to_vvd(*begin);

    if (++begin == end) {
        // Iteratable with only 2 elements.
        pygmo_throw(PyExc_ValueError, "cannot convert an iteratable with only 2 elements into a "
                                      "pagmo::individuals_group_t (exactly 3 elements are needed)");
    }

    // Try fetching the fitness vectors.
    auto fvs_vec = to_vvd(*begin);

    if (++begin != end) {
        // Iteratable with too many elements.
        pygmo_throw(PyExc_ValueError, "cannot convert an iteratable with more than 3 elements into a "
                                      "pagmo::individuals_group_t (exactly 3 elements are needed)");
    }

    return pagmo::individuals_group_t(std::move(ID_vec), std::move(dvs_vec), std::move(fvs_vec));
}

} // namespace pygmo

#endif
