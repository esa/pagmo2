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

#ifndef PAGMO_CUSTOM_COMPARISONS_HPP
#define PAGMO_CUSTOM_COMPARISONS_HPP

#include <algorithm>
#include <boost/functional/hash.hpp> // boost::hash_combine
#include <cstddef>
#include <type_traits>
#include <vector>

#include <pagmo/type_traits.hpp>

namespace pagmo
{
namespace detail
{
// Less than compares floating point types placing nans after inf or before -inf
// It is a useful function when calling e.g. std::sort to guarantee a weak strict ordering
// and avoid an undefined behaviour
template <typename T, bool After = true>
inline bool less_than_f(T a, T b)
{
    static_assert(std::is_floating_point<T>::value, "less_than_f can be used only with floating-point types.");
    if (!std::isnan(a)) {
        if (!std::isnan(b))
            return a < b; // a < b
        else
            return After; // a < nan
    } else {
        if (!std::isnan(b))
            return !After; // nan < b
        else
            return false; // nan < nan
    }
}

// Greater than compares floating point types placing nans after inf or before -inf
// It is a useful function when calling e.g. std::sort to guarantee a weak strict ordering
// and avoid an undefined behaviour
template <typename T, bool After = true>
inline bool greater_than_f(T a, T b)
{
    static_assert(std::is_floating_point<T>::value, "greater_than_f can be used only with floating-point types.");
    if (!std::isnan(a)) {
        if (!std::isnan(b))
            return a > b; // a > b
        else
            return !After; // a > nan
    } else {
        if (!std::isnan(b))
            return After; // nan > b
        else
            return false; // nan > nan
    }
}

// equal_to than compares floating point types considering nan==nan
template <typename T>
inline bool equal_to_f(T a, T b)
{
    static_assert(std::is_floating_point<T>::value, "equal_to_f can be used only with floating-point types.");
    if (!std::isnan(a) && !std::isnan(b)) {
        return a == b;
    }
    return std::isnan(a) && std::isnan(b);
}

// equal_to_vf than compares vectors of floating point types considering nan==nan
template <typename T>
struct equal_to_vf {
    bool operator()(const std::vector<T> &lhs, const std::vector<T> &rhs) const
    {
        static_assert(std::is_floating_point<T>::value,
                      "This class (equal_to_vf) can be used only with floating-point types.");
        if (lhs.size() != rhs.size()) {
            return false;
        } else {
            return std::equal(lhs.begin(), lhs.end(), rhs.begin(), equal_to_f<T>);
        }
    }
};

// hash_vf can be used to hash vectors of floating point types
template <typename T>
struct hash_vf {
    std::size_t operator()(std::vector<T> const &in) const
    {
        static_assert(std::is_floating_point<T>::value,
                      "This class (hash_vf) can be used only with floating-point types.");
        std::size_t retval = 0u;
        for (T el : in) {
            // Combine the hash of the current vector with the hashes of the previous ones
            boost::hash_combine(retval, el);
        }
        return retval;
    }
};
} // namespace detail
} // namespace pagmo

#endif
