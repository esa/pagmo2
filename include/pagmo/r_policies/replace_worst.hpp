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

#ifndef PAGMO_R_POLICIES_REPLACE_WORST_HPP
#define PAGMO_R_POLICIES_REPLACE_WORST_HPP

#include <string>
#include <type_traits>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/variant/variant.hpp>

#include <pagmo/detail/island_fwd.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

class PAGMO_DLL_PUBLIC replace_worst
{
    void verify_fp_ctor() const;

    // Dispatching for the generic ctor
    // via two private constructors: one
    // for absolute migration rate, one
    // for fractional migration rate.
    struct ptag {
    };
    // Absolute migration rate.
    template <typename T, enable_if_t<std::is_integral<T>::value, int> = 0>
    explicit replace_worst(ptag, T n) : m_migr_rate(boost::numeric_cast<population::size_type>(n))
    {
    }
    // Fractional migration rate.
    template <typename T, enable_if_t<std::is_floating_point<T>::value, int> = 0>
    explicit replace_worst(ptag, T x) : m_migr_rate(static_cast<double>(x))
    {
        verify_fp_ctor();
    }

public:
    replace_worst();
    template <typename T,
              enable_if_t<detail::disjunction<std::is_integral<T>, std::is_floating_point<T>>::value, int> = 0>
    explicit replace_worst(T x) : replace_worst(ptag{}, x)
    {
    }

    migrants_t replace(island &, const migrants_t &) const;

    std::string get_name() const
    {
        return "Replace worst";
    }

    std::string get_extra_info() const;

    // Serialization support.
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    boost::variant<population::size_type, double> m_migr_rate;
};

} // namespace pagmo

PAGMO_S11N_R_POLICY_EXPORT_KEY(pagmo::replace_worst)

#endif
