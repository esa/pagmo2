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

#include <cmath>
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/topologies/unconnected.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

namespace detail
{

// Helper to check if w can be used as edge weight
// in a topology.
void topology_check_edge_weight(double w)
{
    if (!std::isfinite(w)) {
        pagmo_throw(std::invalid_argument,
                    "invalid weight for the edge of a topology: the value " + std::to_string(w) + " is not finite");
    }
    if (w < 0. || w > 1.) {
        pagmo_throw(std::invalid_argument, "invalid weight for the edge of a topology: the value " + std::to_string(w)
                                               + " is not in the [0., 1.] range");
    }
}

} // namespace detail

topology::topology() : topology(unconnected{}) {}

void topology::generic_ctor_impl()
{
    // We store at construction the value returned from the user implemented get_name().
    m_name = ptr()->get_name();
}

topology::topology(const topology &other) : m_ptr(other.m_ptr->clone()), m_name(other.m_name) {}

topology::topology(topology &&other) noexcept : m_ptr(std::move(other.m_ptr)), m_name(std::move(other.m_name)) {}

topology &topology::operator=(topology &&other) noexcept
{
    if (this != &other) {
        m_ptr = std::move(other.m_ptr);
        m_name = std::move(other.m_name);
    }
    return *this;
}

topology &topology::operator=(const topology &other)
{
    // Copy ctor + move assignment.
    return *this = topology(other);
}

std::string topology::get_extra_info() const
{
    return ptr()->get_extra_info();
}

bool topology::is_valid() const
{
    return static_cast<bool>(m_ptr);
}

std::pair<std::vector<std::size_t>, vector_double> topology::get_connections(std::size_t n) const
{
    auto retval = ptr()->get_connections(n);

    // Check the returned value.
    if (retval.first.size() != retval.second.size()) {
        pagmo_throw(std::invalid_argument,
                    "An invalid pair of vectors was returned by the 'get_connections()' method of the '" + get_name()
                        + "' topology: the vector of connecting islands has a size of "
                        + std::to_string(retval.first.size())
                        + ", while the vector of migration probabilities has a size of "
                        + std::to_string(retval.second.size()) + " (the two sizes must be equal)");
    }

    for (const auto &p : retval.second) {
        if (!std::isfinite(p)) {
            pagmo_throw(
                std::invalid_argument,
                "An invalid non-finite migration probability of " + std::to_string(p)
                    + " was detected in the vector of migration probabilities returned by the 'get_connections()' "
                      "method of the '"
                    + get_name() + "' topology");
        }
        if (p < 0. || p > 1.) {
            pagmo_throw(
                std::invalid_argument,
                "An invalid migration probability of " + std::to_string(p)
                    + " was detected in the vector of migration probabilities returned by the 'get_connections()' "
                      "method of the '"
                    + get_name() + "' topology: the value must be in the [0., 1.] range");
        }
    }

    return retval;
}

void topology::push_back()
{
    ptr()->push_back();
}

void topology::push_back(unsigned n)
{
    for (auto i = 0u; i < n; ++i) {
        push_back();
    }
}

#if !defined(PAGMO_DOXYGEN_INVOKED)

std::ostream &operator<<(std::ostream &os, const topology &t)
{
    os << "Topology name: " << t.get_name();
    const auto extra_str = t.get_extra_info();
    if (!extra_str.empty()) {
        os << "\nExtra info:\n" << extra_str;
    }
    return os;
}

#endif

} // namespace pagmo
