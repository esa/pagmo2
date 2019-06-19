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

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <pagmo/exceptions.hpp>
#include <pagmo/r_policies/fair_replace.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

// Default constructor: fair_replace with default parameters.
r_policy::r_policy() : r_policy(fair_replace{}) {}

// Implementation of the generic constructor.
void r_policy::generic_ctor_impl()
{
    // Assign the name.
    m_name = ptr()->get_name();
}

// Copy constructor.
r_policy::r_policy(const r_policy &other) : m_ptr(other.ptr()->clone()), m_name(other.m_name) {}

// Move constructor. The default implementation is fine.
r_policy::r_policy(r_policy &&) noexcept = default;

// Move assignment operator
r_policy &r_policy::operator=(r_policy &&other) noexcept
{
    if (this != &other) {
        m_ptr = std::move(other.m_ptr);
        m_name = std::move(other.m_name);
    }
    return *this;
}

// Copy assignment operator
r_policy &r_policy::operator=(const r_policy &other)
{
    // Copy ctor + move assignment.
    return *this = r_policy(other);
}

// Replace individuals in inds with the input migrants mig.
individuals_group_t r_policy::replace(const individuals_group_t &inds, const individuals_group_t &mig) const
{
    // Check the input individuals.
    if (std::get<0>(inds).size() != std::get<1>(inds).size() || std::get<0>(inds).size() != std::get<2>(inds).size()) {
        pagmo_throw(std::invalid_argument,
                    "an invalid group of individuals was passed to a replacement policy of type '" + get_name()
                        + "': the sets of individuals IDs, decision vectors and fitness vectors "
                          "must all have the same sizes, but instead their sizes are "
                        + std::to_string(std::get<0>(inds).size()) + ", " + std::to_string(std::get<1>(inds).size())
                        + " and " + std::to_string(std::get<2>(inds).size()));
    }

    // Check the input migrants.
    if (std::get<0>(mig).size() != std::get<1>(mig).size() || std::get<0>(mig).size() != std::get<2>(mig).size()) {
        pagmo_throw(std::invalid_argument,
                    "an invalid group of migrants was passed to a replacement policy of type '" + get_name()
                        + "': the sets of migrants IDs, decision vectors and fitness vectors "
                          "must all have the same sizes, but instead their sizes are "
                        + std::to_string(std::get<0>(mig).size()) + ", " + std::to_string(std::get<1>(mig).size())
                        + " and " + std::to_string(std::get<2>(mig).size()));
    }

    // Call the replace() method from the UDRP.
    auto retval = ptr()->replace(inds, mig);

    // Check the return value.
    if (std::get<0>(retval).size() != std::get<1>(retval).size()
        || std::get<0>(retval).size() != std::get<2>(retval).size()) {
        pagmo_throw(std::invalid_argument,
                    "an invalid group of individuals was returned by a replacement policy of type '" + get_name()
                        + "': the sets of individuals IDs, decision vectors and fitness vectors "
                          "must all have the same sizes, but instead their sizes are "
                        + std::to_string(std::get<0>(retval).size()) + ", " + std::to_string(std::get<1>(retval).size())
                        + " and " + std::to_string(std::get<2>(retval).size()));
    }

    return retval;
}

// Extra info.
std::string r_policy::get_extra_info() const
{
    return ptr()->get_extra_info();
}

// Check if the r_policy is in a valid state.
bool r_policy::is_valid() const
{
    return static_cast<bool>(m_ptr);
}

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Stream operator.
std::ostream &operator<<(std::ostream &os, const r_policy &r)
{
    os << "Replacement policy name: " << r.get_name() << '\n';
    const auto extra_str = r.get_extra_info();
    if (!extra_str.empty()) {
        os << "\nExtra info:\n" << extra_str << '\n';
    }
    return os;
}

#endif

} // namespace pagmo
