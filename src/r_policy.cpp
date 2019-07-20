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

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <pagmo/exceptions.hpp>
#include <pagmo/r_policies/fair_replace.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/types.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

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

// Verify the input arguments for the replace() function.
void r_policy::verify_replace_input(const individuals_group_t &inds, const vector_double::size_type &nx,
                                    const vector_double::size_type &nix, const vector_double::size_type &nobj,
                                    const vector_double::size_type &nec, const vector_double::size_type &nic,
                                    const vector_double &tol, const individuals_group_t &mig) const
{
    // 1 - verify that the elements of inds all have the same size.
    if (std::get<0>(inds).size() != std::get<1>(inds).size() || std::get<0>(inds).size() != std::get<2>(inds).size()) {
        pagmo_throw(std::invalid_argument,
                    "an invalid group of individuals was passed to a replacement policy of type '" + get_name()
                        + "': the sets of individuals IDs, decision vectors and fitness vectors "
                          "must all have the same sizes, but instead their sizes are "
                        + std::to_string(std::get<0>(inds).size()) + ", " + std::to_string(std::get<1>(inds).size())
                        + " and " + std::to_string(std::get<2>(inds).size()));
    }

    // 2 - same for mig.
    if (std::get<0>(mig).size() != std::get<1>(mig).size() || std::get<0>(mig).size() != std::get<2>(mig).size()) {
        pagmo_throw(std::invalid_argument,
                    "an invalid group of migrants was passed to a replacement policy of type '" + get_name()
                        + "': the sets of migrants IDs, decision vectors and fitness vectors "
                          "must all have the same sizes, but instead their sizes are "
                        + std::to_string(std::get<0>(mig).size()) + ", " + std::to_string(std::get<1>(mig).size())
                        + " and " + std::to_string(std::get<2>(mig).size()));
    }

    // 3 - make sure nx, nix, nobj, nec, nic are sane and consistent.
    // Check that the problem dimension is not zero.
    if (!nx) {
        pagmo_throw(std::invalid_argument,
                    "a problem dimension of zero was passed to a replacement policy of type '" + get_name() + "'");
    }
    // Verify that it is consistent with nix.
    if (nix > nx) {
        pagmo_throw(std::invalid_argument,
                    "the integer dimension (" + std::to_string(nix) + ") passed to a replacement policy of type '"
                        + get_name() + "' is larger than the supplied problem dimension (" + std::to_string(nx) + ")");
    }
    if (!nobj) {
        pagmo_throw(std::invalid_argument,
                    "an invalid number of objectives (0) was passed to a replacement policy of type '" + get_name()
                        + "'");
    }
    if (nobj > std::numeric_limits<vector_double::size_type>::max() / 3u) {
        pagmo_throw(std::invalid_argument, "the number of objectives (" + std::to_string(nobj)
                                               + ") passed to a replacement policy of type '" + get_name()
                                               + "' is too large");
    }
    if (nec > std::numeric_limits<vector_double::size_type>::max() / 3u) {
        pagmo_throw(std::invalid_argument, "the number of equality constraints (" + std::to_string(nec)
                                               + ") passed to a replacement policy of type '" + get_name()
                                               + "' is too large");
    }
    if (nic > std::numeric_limits<vector_double::size_type>::max() / 3u) {
        pagmo_throw(std::invalid_argument, "the number of inequality constraints (" + std::to_string(nic)
                                               + ") passed to a replacement policy of type '" + get_name()
                                               + "' is too large");
    }
    // Verify that the tol vector size is correct.
    if (tol.size() != nec + nic) {
        pagmo_throw(std::invalid_argument, "the vector of tolerances passed to a replacement policy of type '"
                                               + get_name() + "' has a dimension (" + std::to_string(tol.size())
                                               + ") which is inconsistent with the total number of constraints ("
                                               + std::to_string(nec + nic) + ")");
    }
    // Determine the fitness dimension.
    const auto nf = nobj + nec + nic;

    // 4 - verify inds/migs.
    auto dv_checker = [nx](const vector_double &dv) { return dv.size() != nx; };
    auto fv_checker = [nf](const vector_double &fv) { return fv.size() != nf; };

    if (std::any_of(std::get<1>(inds).begin(), std::get<1>(inds).end(), dv_checker)) {
        pagmo_throw(std::invalid_argument, "not all the individuals passed to a replacement policy of type '"
                                               + get_name() + "' have the expected dimension (" + std::to_string(nx)
                                               + ")");
    }
    if (std::any_of(std::get<2>(inds).begin(), std::get<2>(inds).end(), fv_checker)) {
        pagmo_throw(std::invalid_argument, "not all the individuals passed to a replacement policy of type '"
                                               + get_name() + "' have the expected fitness dimension ("
                                               + std::to_string(nf) + ")");
    }
    if (std::any_of(std::get<1>(mig).begin(), std::get<1>(mig).end(), dv_checker)) {
        pagmo_throw(std::invalid_argument, "not all the migrants passed to a replacement policy of type '" + get_name()
                                               + "' have the expected dimension (" + std::to_string(nx) + ")");
    }
    if (std::any_of(std::get<2>(mig).begin(), std::get<2>(mig).end(), fv_checker)) {
        pagmo_throw(std::invalid_argument, "not all the migrants passed to a replacement policy of type '" + get_name()
                                               + "' have the expected fitness dimension (" + std::to_string(nf) + ")");
    }
}

// Verify the output of replace().
void r_policy::verify_replace_output(const individuals_group_t &retval, vector_double::size_type nx,
                                     vector_double::size_type nf) const
{
    // 1 - verify that the elements of retval all have the same size.
    if (std::get<0>(retval).size() != std::get<1>(retval).size()
        || std::get<0>(retval).size() != std::get<2>(retval).size()) {
        pagmo_throw(std::invalid_argument,
                    "an invalid group of individuals was returned by a replacement policy of type '" + get_name()
                        + "': the sets of individuals IDs, decision vectors and fitness vectors "
                          "must all have the same sizes, but instead their sizes are "
                        + std::to_string(std::get<0>(retval).size()) + ", " + std::to_string(std::get<1>(retval).size())
                        + " and " + std::to_string(std::get<2>(retval).size()));
    }

    // 2 - verify that the decision/fitness vectors in retval have all
    // the expected dimensions.
    if (std::any_of(std::get<1>(retval).begin(), std::get<1>(retval).end(),
                    [nx](const vector_double &dv) { return dv.size() != nx; })) {
        pagmo_throw(std::invalid_argument, "not all the individuals returned by a replacement policy of type '"
                                               + get_name() + "' have the expected dimension (" + std::to_string(nx)
                                               + ")");
    }
    if (std::any_of(std::get<2>(retval).begin(), std::get<2>(retval).end(),
                    [nf](const vector_double &fv) { return fv.size() != nf; })) {
        pagmo_throw(std::invalid_argument, "not all the individuals returned by a replacement policy of type '"
                                               + get_name() + "' have the expected fitness dimension ("
                                               + std::to_string(nf) + ")");
    }
}

// Replace individuals in inds with the input migrants mig.
individuals_group_t r_policy::replace(const individuals_group_t &inds, const vector_double::size_type &nx,
                                      const vector_double::size_type &nix, const vector_double::size_type &nobj,
                                      const vector_double::size_type &nec, const vector_double::size_type &nic,
                                      const vector_double &tol, const individuals_group_t &mig) const
{
    // Verify the input.
    verify_replace_input(inds, nx, nix, nobj, nec, nic, tol, mig);

    // Call the replace() method from the UDRP.
    auto retval = ptr()->replace(inds, nx, nix, nobj, nec, nic, tol, mig);

    // Verify the output.
    // NOTE: we checked in verify_replace_input() that we can
    // compute nobj + nec + nic safely.
    verify_replace_output(retval, nx, nobj + nec + nic);

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
