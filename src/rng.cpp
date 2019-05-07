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

#include <mutex>
#include <random>

#include <pagmo/rng.hpp>

namespace pagmo
{

namespace detail
{

namespace
{

// The global rng is inited on startup with a random number.
random_engine_type global_rng(static_cast<random_engine_type::result_type>(std::random_device()()));

std::mutex global_rng_mutex;

} // namespace

} // namespace detail

/// Next element of the Pseudo Random Sequence
/**
 * This static method returns the next element of the PRS.
 *
 * @returns the next element of the PRS
 */
unsigned random_device::next()
{
    std::lock_guard<std::mutex> lock(detail::global_rng_mutex);
    return static_cast<unsigned>(detail::global_rng());
}

/// Sets the seed for the PRS
/**
 * This static method sets a new seed for the PRS, so that all the
 * following calls to random_device::next() will always repeat the same
 * numbers.
 *
 * @param seed The new seed to be used
 */
void random_device::set_seed(unsigned seed)
{
    std::lock_guard<std::mutex> lock(detail::global_rng_mutex);
    detail::global_rng.seed(static_cast<detail::random_engine_type::result_type>(seed));
}

} // namespace pagmo
