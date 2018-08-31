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

#ifndef PAGMO_SERIALIZATION_HPP
#define PAGMO_SERIALIZATION_HPP

// Let's disable a few compiler warnings emitted by the cereal code.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
// NOTE: these warnings are available on all the supported versions
// of GCC/clang, no need to put version checks.
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wdeprecated"
// MINGW-specific warnings.
#if defined(__MINGW32__)
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif
#if __GNUC__ >= 7
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#pragma GCC diagnostic ignored "-Wnoexcept"
#pragma GCC diagnostic ignored "-Wcast-align"
#endif
#if defined(__clang__)

#if defined(__apple_build_version__)

// LLVM 3.2 -> Xcode 4.6.
#if __clang_major__ > 4 || (__clang_major__ == 4 && __clang_minor__ >= 6)

#pragma GCC diagnostic ignored "-Wunused-private-field"

#endif

// LLVM 3.7 -> Xcode 7.0.
#if __clang_major__ >= 7

#pragma GCC diagnostic ignored "-Wexceptions"

#endif

#else

// LLVM 3.2.
#if __clang_major__ > 3 || (__clang_major__ == 3 && __clang_minor__ >= 2)

#pragma GCC diagnostic ignored "-Wunused-private-field"

#endif

// LLVM 3.7.
#if __clang_major__ > 3 || (__clang_major__ == 3 && __clang_minor__ >= 7)

#pragma GCC diagnostic ignored "-Wexceptions"

#endif

#endif

#endif
#endif

// Enable thread-safety in cereal. See:
// http://uscilab.github.io/cereal/thread_safety.html
#define CEREAL_THREAD_SAFE 1

// Types first.
#include <pagmo/external/cereal/types/base_class.hpp>
#include <pagmo/external/cereal/types/common.hpp>
#include <pagmo/external/cereal/types/map.hpp>
#include <pagmo/external/cereal/types/memory.hpp>
#include <pagmo/external/cereal/types/polymorphic.hpp>
#include <pagmo/external/cereal/types/tuple.hpp>
#include <pagmo/external/cereal/types/utility.hpp>
#include <pagmo/external/cereal/types/vector.hpp>

// Then the archives.
#include <pagmo/external/cereal/archives/binary.hpp>
#include <pagmo/external/cereal/archives/json.hpp>
#include <pagmo/external/cereal/archives/portable_binary.hpp>

#undef CEREAL_THREAD_SAFE

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <pagmo/config.hpp>

#include <cstddef>
#include <random>
#include <sstream>
#include <string>

#if defined(PAGMO_WITH_EIGEN3)
#include <pagmo/detail/eigen.hpp>
#endif

namespace cereal
{
// Implement serialization for the Mersenne twister engine.
template <class Archive, class UIntType, std::size_t w, std::size_t n, std::size_t m, std::size_t r, UIntType a,
          std::size_t u, UIntType d, std::size_t s, UIntType b, std::size_t t, UIntType c, std::size_t l, UIntType f>
inline void
CEREAL_SAVE_FUNCTION_NAME(Archive &ar,
                          std::mersenne_twister_engine<UIntType, w, n, m, r, a, u, d, s, b, t, c, l, f> const &e)
{
    std::ostringstream oss;
    // Use the "C" locale.
    oss.imbue(std::locale::classic());
    oss << e;
    ar(oss.str());
}
template <class Archive, class UIntType, std::size_t w, std::size_t n, std::size_t m, std::size_t r, UIntType a,
          std::size_t u, UIntType d, std::size_t s, UIntType b, std::size_t t, UIntType c, std::size_t l, UIntType f>
inline void CEREAL_LOAD_FUNCTION_NAME(Archive &ar,
                                      std::mersenne_twister_engine<UIntType, w, n, m, r, a, u, d, s, b, t, c, l, f> &e)
{
    std::istringstream iss;
    // Use the "C" locale.
    iss.imbue(std::locale::classic());
    std::string tmp;
    ar(tmp);
    iss.str(tmp);
    iss >> e;
}

#if defined(PAGMO_WITH_EIGEN3)
// Implement the serialization of the Eigen::Matrix class
template <class Archive, class S, int R, int C, int O, int MR, int MC>
inline void CEREAL_SAVE_FUNCTION_NAME(Archive &ar, Eigen::Matrix<S, R, C, O, MR, MC> const &cb)
{
    // Let's first save the dimension
    auto nrows = cb.rows();
    auto ncols = cb.cols();
    ar << nrows;
    ar << ncols;
    // And then the numbers
    for (decltype(nrows) i = 0; i < nrows; ++i) {
        for (decltype(nrows) j = 0; j < ncols; ++j) {
            ar << cb(i, j);
        }
    }
}
template <class Archive, class S, int R, int C, int O, int MR, int MC>
inline void CEREAL_LOAD_FUNCTION_NAME(Archive &ar, Eigen::Matrix<S, R, C, O, MR, MC> &cb)
{
    decltype(cb.rows()) nrows;
    decltype(cb.cols()) ncols;
    // Let's first restore the dimension
    ar >> nrows;
    ar >> ncols;
    cb.resize(nrows, ncols);
    // And then the numbers
    for (decltype(nrows) i = 0; i < nrows; ++i) {
        for (decltype(nrows) j = 0; j < ncols; ++j) {
            ar >> cb(i, j);
        }
    }
}
#endif
} // namespace cereal

#endif
