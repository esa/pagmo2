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

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

#include <boost/variant/get.hpp>
#include <boost/variant/variant.hpp>

#include <pagmo/detail/base_sr_policy.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=const"
#endif

namespace pagmo
{

namespace detail
{

// Helper to verify the ctor from a fractional rate.
void base_sr_policy::verify_fp_ctor() const
{
    assert(m_migr_rate.which() == 1);

    const auto rate = boost::get<double>(m_migr_rate);

    if (!std::isfinite(rate) || rate < 0. || rate > 1.) {
        pagmo_throw(std::invalid_argument,
                    "Invalid fractional migration rate specified in the constructor of a replacement/selection "
                    "policy: the rate must be in the [0., 1.] range, but it is "
                        + std::to_string(rate) + " instead");
    }
}

// Getter for the migration rate variant.
const boost::variant<pop_size_t, double> &base_sr_policy::get_migr_rate() const
{
    return m_migr_rate;
}

} // namespace detail

} // namespace pagmo
