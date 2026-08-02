/* Copyright 2017-2021 PaGMO development team

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

#ifndef PAGMO_DETAIL_REFERENCE_POINT_HPP
#define PAGMO_DETAIL_REFERENCE_POINT_HPP

#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp>  // PAGMO_DLL_PUBLIC
#include <pagmo/population.hpp>         // pop_size_t
#include <pagmo/rng.hpp>                // random_engine_type

// NOTE: this header defines the reference point machinery of the NSGA-III
// algorithm. It is an implementation detail of pagmo::nsga3.
namespace pagmo
{

namespace detail
{

class PAGMO_DLL_PUBLIC reference_point{
    public:
        reference_point(size_t nobj);
        size_t dim() const;
        double& operator[](size_t);
        friend PAGMO_DLL_PUBLIC std::ostream& operator<<(std::ostream& ostr, const reference_point& rp);
        void increment_members(){ ++nmembers; }
        void decrement_members(){ --nmembers; }
        size_t member_count() const{ return nmembers; }
        void add_candidate(size_t, double);
        void remove_candidate(size_t index);
        size_t candidate_count() const{ return candidates.size(); }
        const std::vector<double> &get_coeffs() const{ return coeffs; }
        std::optional<size_t> nearest_candidate() const;
        std::optional<size_t> random_candidate(random_engine_type &) const;
        std::optional<size_t> select_member(random_engine_type &) const;
    protected:
        std::vector<double> coeffs{0};
        size_t nmembers{0};
        std::vector<std::pair<size_t, double>> candidates;
};

PAGMO_DLL_PUBLIC std::vector<reference_point> generate_reference_point_level(
    reference_point& rp,
    size_t remain,
    size_t level,
    size_t total
);

// Structured set of reference points on the unit simplex, Deb & Jain Section IV.A
PAGMO_DLL_PUBLIC std::vector<reference_point> generate_uniform_reference_points(size_t nobjs, size_t divisions);

PAGMO_DLL_PUBLIC void associate_with_reference_points(
    std::vector<reference_point> &,                 // Reference points
    const std::vector<std::vector<double>> &,       // Normalized objectives
    const std::vector<std::vector<pop_size_t>> &    // NDS Fronts
);

PAGMO_DLL_PUBLIC size_t identify_niche_point(std::vector<reference_point> &, random_engine_type &);

PAGMO_DLL_PUBLIC size_t n_choose_k(size_t, size_t);

} // namespace detail

} // namespace pagmo

#endif
