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

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include <pagmo/detail/time_limit.hpp>
#include <pagmo/exceptions.hpp>

namespace pagmo
{

namespace detail
{

void check_max_time(double max_time)
{
    if (max_time < 0. || (std::isfinite(max_time) == false)) {
        pagmo_throw(std::invalid_argument,
                    "The maximum time must be a non-negative finite number of seconds, while a value of "
                        + std::to_string(max_time) + " was detected.");
    }
}

time_limit::time_limit(double max_time) : m_max_time(max_time), m_start(std::chrono::steady_clock::now()) {}

bool time_limit::is_reached() const
{
    // NOTE: the elapsed time is compared as a floating-point number of seconds, which
    // avoids overflowing the integral representation of steady_clock for huge limits.
    return m_max_time > 0.
           && std::chrono::duration<double>(std::chrono::steady_clock::now() - m_start).count() >= m_max_time;
}

} // namespace detail

} // namespace pagmo
