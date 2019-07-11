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

#ifndef PAGMO_DETAIL_BASE_SR_POLICY_HPP
#define PAGMO_DETAIL_BASE_SR_POLICY_HPP

#include <type_traits>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/variant/variant.hpp>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

class PAGMO_DLL_PUBLIC base_sr_policy
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
    explicit base_sr_policy(ptag, T n) : m_migr_rate(boost::numeric_cast<pop_size_t>(n))
    {
    }
    // Fractional migration rate.
    template <typename T, enable_if_t<std::is_floating_point<T>::value, int> = 0>
    explicit base_sr_policy(ptag, T x) : m_migr_rate(static_cast<double>(x))
    {
        verify_fp_ctor();
    }

public:
    // Constructor from fractional or absolute migration policy.
    template <typename T,
              enable_if_t<detail::disjunction<std::is_integral<T>, std::is_floating_point<T>>::value, int> = 0>
    explicit base_sr_policy(T x) : base_sr_policy(ptag{}, x)
    {
    }

    // Serialization support.
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        detail::archive(ar, m_migr_rate);
    }

    const boost::variant<pop_size_t, double> &get_migr_rate() const;

protected:
    boost::variant<pop_size_t, double> m_migr_rate;
};

} // namespace detail

} // namespace pagmo

// Disable tracking for the serialisation of base_sr_policy.
BOOST_CLASS_TRACKING(pagmo::detail::base_sr_policy, boost::serialization::track_never)

#endif
