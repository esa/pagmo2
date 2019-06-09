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
#include <string>
#include <utility>

#include <pagmo/batch_evaluators/default_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/bfe_impl.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

// Default ctor.
bfe::bfe() : bfe(default_bfe{}) {}

// Implementation of the generic ctor.
void bfe::generic_ctor_impl()
{
    // Assign the name.
    m_name = ptr()->get_name();
    // Assign the thread safety level.
    m_thread_safety = ptr()->get_thread_safety();
}

// Copy constructor.
bfe::bfe(const bfe &other) : m_ptr(other.ptr()->clone()), m_name(other.m_name), m_thread_safety(other.m_thread_safety)
{
}

// Move constructor. The default implementation is fine.
bfe::bfe(bfe &&) noexcept = default;

// Move assignment operator
bfe &bfe::operator=(bfe &&other) noexcept
{
    if (this != &other) {
        m_ptr = std::move(other.m_ptr);
        m_name = std::move(other.m_name);
        m_thread_safety = std::move(other.m_thread_safety);
    }
    return *this;
}

// Copy assignment operator
bfe &bfe::operator=(const bfe &other)
{
    // Copy ctor + move assignment.
    return *this = bfe(other);
}

// Call operator.
vector_double bfe::operator()(const problem &p, const vector_double &dvs) const
{
    // Check the input dvs.
    detail::bfe_check_input_dvs(p, dvs);
    // Invoke the call operator from the UDBFE.
    auto retval((*ptr())(p, dvs));
    // Check the produced vector of fitnesses.
    detail::bfe_check_output_fvs(p, dvs, retval);
    return retval;
}

// Extra info.
std::string bfe::get_extra_info() const
{
    return ptr()->get_extra_info();
}

// Check if the bfe is in a valid state.
bool bfe::is_valid() const
{
    return static_cast<bool>(m_ptr);
}

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Stream operator.
std::ostream &operator<<(std::ostream &os, const bfe &b)
{
    os << "BFE name: " << b.get_name() << '\n';
    os << "\n\tThread safety: " << b.get_thread_safety() << '\n';
    const auto extra_str = b.get_extra_info();
    if (!extra_str.empty()) {
        os << "\nExtra info:\n" << extra_str << '\n';
    }
    return os;
}

#endif

} // namespace pagmo
