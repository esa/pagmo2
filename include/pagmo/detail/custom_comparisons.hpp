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

#ifndef PAGMO_CUSTOM_COMPARISONS_HPP
#define PAGMO_CUSTOM_COMPARISONS_HPP

#include <type_traits>

#include "../type_traits.hpp"

    namespace pagmo
{
    namespace detail
    {
    // Less than compares floating point types placing nans after inf or before -inf
    // It is a useful function when calling e.g. std::sort to guarantee a weak strict ordering
    // and avoid an undefined behaviour
    template <typename T, bool After = true, enable_if_is_floating_point<T> = 0>
    inline bool less_than_f(T a, T b)
    {
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
    template <typename T, bool After = true, detail::enable_if_is_floating_point<T> = 0>
    inline bool greater_than_f(T a, T b)
    {
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
    template <typename T, detail::enable_if_is_floating_point<T> = 0>
    inline bool equal_to_f(T a, T b)
    {
        if (!std::isnan(a) && !std::isnan(b)) {
            return a == b;
        }
        return std::isnan(a) && std::isnan(b);
    }

    } // end of detail namespace
} // end of pagmo namespace

#endif
