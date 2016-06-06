#ifndef PYGMO_NUMPY_HPP
#define PYGMO_NUMPY_HPP

#include "python_includes.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-function"
#endif

#include <numpy/arrayobject.h>

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#undef NPY_NO_DEPRECATED_API

#endif
