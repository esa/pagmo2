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

#include <boost/serialization/variant.hpp>
#include <boost/variant/get.hpp>

#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/r_policies/fair_replace.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

// Default constructor: fractional migration rate, 10%.
fair_replace::fair_replace() : fair_replace(.1) {}

// Helper to verify the ctor from a fractional rate.
void fair_replace::verify_fp_ctor() const
{
    assert(m_migr_rate.which() == 1);

    const auto rate = boost::get<double>(m_migr_rate);

    if (!std::isfinite(rate) || rate < 0. || rate > 1.) {
        pagmo_throw(std::invalid_argument,
                    "Invalid fractional migration rate specified in the constructor of the fair_replace replacement "
                    "policy: the rate must be in the [0., 1.] range, but it is "
                        + std::to_string(rate) + " instead");
    }
}

individuals_group_t fair_replace::replace(const individuals_group_t &, const individuals_group_t &) const
{
    return individuals_group_t{};
}

// Extra info.
std::string fair_replace::get_extra_info() const
{
    if (m_migr_rate.which()) {
        const auto rate = boost::get<double>(m_migr_rate);
        return "\tFractional migration rate: " + std::to_string(rate);
    } else {
        const auto rate = boost::get<population::size_type>(m_migr_rate);
        return "\tAbsolute migration rate: " + std::to_string(rate);
    }
}

// Serialization support.
template <typename Archive>
void fair_replace::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_migr_rate);
}

} // namespace pagmo

PAGMO_S11N_R_POLICY_IMPLEMENT(pagmo::fair_replace)
