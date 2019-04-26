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

#ifndef PAGMO_S11N_HPP
#define PAGMO_S11N_HPP

#include <cstddef>
#include <locale>
#include <random>
#include <sstream>
#include <string>
#include <tuple>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

// Include the archives.
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <pagmo/detail/s11n_wrappers.hpp>

namespace pagmo
{

namespace detail
{

// Implementation of tuple serialization.
template <std::size_t N>
struct tuple_s11n {
    template <class Archive, typename... Args>
    static void serialize(Archive &ar, std::tuple<Args...> &t, unsigned version)
    {
        ar &std::get<N - 1u>(t);
        tuple_s11n<N - 1u>::serialize(ar, t, version);
    }
};

template <>
struct tuple_s11n<0> {
    template <class Archive, typename... Args>
    static void serialize(Archive &, std::tuple<Args...> &, unsigned)
    {
    }
};

} // namespace detail

} // namespace pagmo

namespace boost
{

namespace serialization
{

// Implement serialization for std::tuple.
template <class Archive, typename... Args>
void serialize(Archive &ar, std::tuple<Args...> &t, unsigned version)
{
    pagmo::detail::tuple_s11n<sizeof...(Args)>::serialize(ar, t, version);
}

// Implement serialization for the Mersenne twister engine.
template <class Archive, class UIntType, std::size_t w, std::size_t n, std::size_t m, std::size_t r, UIntType a,
          std::size_t u, UIntType d, std::size_t s, UIntType b, std::size_t t, UIntType c, std::size_t l, UIntType f>
inline void save(Archive &ar, std::mersenne_twister_engine<UIntType, w, n, m, r, a, u, d, s, b, t, c, l, f> const &e,
                 unsigned)
{
    std::ostringstream oss;
    // Use the "C" locale.
    oss.imbue(std::locale::classic());
    oss << e;
    ar << oss.str();
}

template <class Archive, class UIntType, std::size_t w, std::size_t n, std::size_t m, std::size_t r, UIntType a,
          std::size_t u, UIntType d, std::size_t s, UIntType b, std::size_t t, UIntType c, std::size_t l, UIntType f>
inline void load(Archive &ar, std::mersenne_twister_engine<UIntType, w, n, m, r, a, u, d, s, b, t, c, l, f> &e,
                 unsigned)
{
    std::istringstream iss;
    // Use the "C" locale.
    iss.imbue(std::locale::classic());
    std::string tmp;
    ar >> tmp;
    iss.str(tmp);
    iss >> e;
}

template <class Archive, class UIntType, std::size_t w, std::size_t n, std::size_t m, std::size_t r, UIntType a,
          std::size_t u, UIntType d, std::size_t s, UIntType b, std::size_t t, UIntType c, std::size_t l, UIntType f>
inline void serialize(Archive &ar, std::mersenne_twister_engine<UIntType, w, n, m, r, a, u, d, s, b, t, c, l, f> &e,
                      unsigned version)
{
    split_free(ar, e, version);
}

} // namespace serialization

} // namespace boost

#endif
