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

#ifndef PYGMO_DEMANGLE_HPP
#define PYGMO_DEMANGLE_HPP

#include <boost/version.hpp>
#include <string>
#include <typeindex>
#include <typeinfo>

#if BOOST_VERSION / 100000 > 1 || (BOOST_VERSION / 100000 == 1 && BOOST_VERSION / 100 % 1000 >= 56)

// Boost demangle is available since 1.56.
#include <boost/core/demangle.hpp>
#define PYGMO_HAVE_BOOST_DEMANGLE

#elif defined(__GNUC__)

// GCC demangle. This is available also for clang, both with libstdc++ and libc++.
#include <cstdlib>
#include <cxxabi.h>
#include <memory>

#endif

namespace pygmo
{

inline std::string demangle(const char *s)
{
#if defined(PYGMO_HAVE_BOOST_DEMANGLE)
    return boost::core::demangle(s);
#undef PYGMO_HAVE_BOOST_DEMANGLE
#elif defined(__GNUC__)
    int status = -4;
    // NOTE: abi::__cxa_demangle will return a pointer allocated by std::malloc, which we will delete via std::free.
    std::unique_ptr<char, void (*)(void *)> res{::abi::__cxa_demangle(s, nullptr, nullptr, &status), std::free};
    // NOTE: it seems like clang with libc++ does not set the status variable properly.
    // We then check if anything was allocated by __cxa_demangle(), as here it mentions
    // that in case of failure the pointer will be set to null:
    // https://gcc.gnu.org/onlinedocs/libstdc++/libstdc++-html-USERS-4.3/a01696.html
    return res.get() ? std::string(res.get()) : std::string(s);
#else
    // TODO demangling for other platforms. E.g.,
    // http://stackoverflow.com/questions/13777681/demangling-in-msvc
    // NOTE: it seems that the Boost implementation currently covers
    // only GCC/Clang. So there might be value in having an MSVC demangler
    // (eventually) and using it even if the Boost one is available.
    return std::string(s);
#endif
}

// C++ string overload.
inline std::string demangle(const std::string &s)
{
    return demangle(s.c_str());
}

// Convenience overload for demangling type_index. Will also work with type_info
// due to implicit conversion.
inline std::string demangle(const std::type_index &t_idx)
{
    return demangle(t_idx.name());
}

// Convenience overload with template type.
template <typename T>
inline std::string demangle()
{
    return demangle(typeid(T));
}
}

#endif
