/* Copyright 2017-2026 PaGMO development team

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

#ifndef PAGMO_DETAIL_TIME_LIMIT_HPP
#define PAGMO_DETAIL_TIME_LIMIT_HPP

#include <chrono>

#include <pagmo/detail/visibility.hpp>

namespace pagmo
{

namespace detail
{

// Validate the max_time argument of the UDAs: it must be a finite, non-negative
// number of seconds (zero disables the time limit).
PAGMO_DLL_PUBLIC void check_max_time(double);

// The wall-clock budget of a single evolve() call, measured with std::chrono::steady_clock
// from the moment of construction. A max_time of zero disables the limit.
class PAGMO_DLL_PUBLIC time_limit
{
public:
    explicit time_limit(double);
    bool is_reached() const;

private:
    double m_max_time;
    std::chrono::steady_clock::time_point m_start;
};

} // namespace detail

} // namespace pagmo

#endif
