#ifndef PYGMO_PYBIND11_HPP
#define PYGMO_PYBIND11_HPP

// Just a utility header to wrap the inclusion of pybind11.

// The pybind11 code produces some warning messages when we compile
// in debug mode.
#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpedantic"
    #pragma GCC diagnostic ignored "-Wshadow"
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wdeprecated"
#endif

#include "../include/external/pybind11/include/pybind11/numpy.h"
#include "../include/external/pybind11/include/pybind11/pybind11.h"
#include "../include/external/pybind11/include/pybind11/stl.h"

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#endif
