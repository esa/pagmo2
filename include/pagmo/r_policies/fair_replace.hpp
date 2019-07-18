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

#ifndef PAGMO_R_POLICIES_FAIR_REPLACE_HPP
#define PAGMO_R_POLICIES_FAIR_REPLACE_HPP

#include <string>
#include <type_traits>

#include <pagmo/detail/base_sr_policy.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

class PAGMO_DLL_PUBLIC fair_replace : public detail::base_sr_policy
{
public:
    // Default ctor.
    fair_replace();

    // Constructor from migration rate.
    template <typename T,
              enable_if_t<detail::disjunction<std::is_integral<T>, std::is_floating_point<T>>::value, int> = 0>
    explicit fair_replace(T x) : detail::base_sr_policy(x)
    {
    }

    // Replacement.
    individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double &, const individuals_group_t &) const;

    std::string get_name() const
    {
        return "Fair replace";
    }
    std::string get_extra_info() const;

    // Serialization support.
    template <typename Archive>
    void serialize(Archive &, unsigned);
};

} // namespace pagmo

PAGMO_S11N_R_POLICY_EXPORT_KEY(pagmo::fair_replace)

#endif
