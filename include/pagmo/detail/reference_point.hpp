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

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp> // PAGMO_DLL_PUBLIC
#include <pagmo/population.hpp>        // pop_size_t
#include <pagmo/rng.hpp>               // random_engine_type

// NOTE: this header defines the reference point machinery of the NSGA-III
// algorithm. It is an implementation detail of pagmo::nsga3.
namespace pagmo
{

namespace detail
{

/*  A structured reference direction on the unit simplex, together with the niching
 *  bookkeeping of Deb & Jain Algorithms 3 and 4: the number of members already
 *  associated with it, and the candidates of the splitting front which may still
 *  be selected for it.
 */
class PAGMO_DLL_PUBLIC reference_point
{
public:
    explicit reference_point(std::size_t nobjs);
    std::size_t dim() const;
    double &operator[](std::size_t);
    double operator[](std::size_t) const;
    friend PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &ostr, const reference_point &rp);
    void increment_members()
    {
        ++nmembers;
    }
    void decrement_members()
    {
        --nmembers;
    }
    std::size_t member_count() const
    {
        return nmembers;
    }
    void add_candidate(std::size_t, double);
    void remove_candidate(std::size_t index);
    std::size_t candidate_count() const
    {
        return candidates.size();
    }
    const std::vector<double> &get_coeffs() const
    {
        return coeffs;
    }
    /*  Drops the generation-specific bookkeeping while retaining the direction
     *  itself, so that an immutable set of directions built once per evolve() can
     *  be reused by every generation.
     */
    void reset()
    {
        nmembers = 0u;
        candidates.clear();
    }
    std::optional<std::size_t> nearest_candidate() const;
    std::optional<std::size_t> random_candidate(random_engine_type &) const;
    std::optional<std::size_t> select_member(random_engine_type &) const;

protected:
    std::vector<double> coeffs;
    std::size_t nmembers{0};
    std::vector<std::pair<std::size_t, double>> candidates;
};

/*  Upper bound on the number of reference directions that will be generated.
 *
 *  The Das and Dennis construction grows combinatorially: Deb & Jain note that
 *  eight objectives with p = 8 already require 5040 directions, which is why they
 *  introduce the two layer scheme. This cap turns a configuration which would
 *  exhaust memory into an informative exception raised before anything is
 *  allocated.
 */
inline constexpr std::size_t max_reference_directions = 100000u;

// Exact binomial coefficient. Throws std::overflow_error if it does not fit in std::size_t.
PAGMO_DLL_PUBLIC std::size_t n_choose_k(std::size_t n, std::size_t k);

// Number of directions of a single Das and Dennis layer, i.e. n_choose_k(nobjs + divisions - 1, divisions)
PAGMO_DLL_PUBLIC std::size_t reference_point_count(std::size_t nobjs, std::size_t divisions);

// A single Das and Dennis layer on the unit simplex, Deb & Jain Section IV.B
PAGMO_DLL_PUBLIC std::vector<reference_point> generate_uniform_reference_points(std::size_t nobjs,
                                                                                std::size_t divisions);

/*  The two layer reference direction set of Deb & Jain Section V.
 *
 *  The outer (boundary) layer is a plain Das and Dennis layer with divisions_outer
 *  divisions. The inner layer is a Das and Dennis layer with divisions_inner
 *  divisions, every coordinate of which is mapped through
 *
 *      c -> (c + 1/nobjs)/2
 *
 *  which shrinks the layer by one half about the centroid of the simplex. The
 *  transformation preserves the simplex constraint, since the coordinates of the
 *  untransformed layer sum to one and there are nobjs of them.
 *
 *  divisions_inner == 0 disables the inner layer. The two layers are concatenated
 *  in that order, each in generation order, and an inner direction which coincides
 *  with one already present is dropped, so the result is deterministic and free of
 *  duplicates.
 */
PAGMO_DLL_PUBLIC std::vector<reference_point>
generate_reference_directions(std::size_t nobjs, std::size_t divisions_outer, std::size_t divisions_inner);

PAGMO_DLL_PUBLIC void associate_with_reference_points(
    std::vector<reference_point> &,              // Reference points
    const std::vector<std::vector<double>> &,    // Normalized objectives
    const std::vector<std::vector<pop_size_t>> & // NDS fronts of S_t, the last one being the splitting front
);

PAGMO_DLL_PUBLIC std::size_t identify_niche_point(std::vector<reference_point> &, random_engine_type &);

} // namespace detail

} // namespace pagmo

#endif
