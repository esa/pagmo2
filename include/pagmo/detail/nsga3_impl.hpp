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

#ifndef PAGMO_DETAIL_NSGA3_IMPL_HPP
#define PAGMO_DETAIL_NSGA3_IMPL_HPP

#include <cstddef>
#include <optional>
#include <random>
#include <vector>

#include <pagmo/detail/reference_point.hpp> // reference_point
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp> // pop_size_t
#include <pagmo/rng.hpp>        // random_engine_type
#include <pagmo/types.hpp>

// NOTE: this header contains the numerical utilities and the adaptive normalisation
// pipeline of the NSGA-III algorithm. They are implementation details of pagmo::nsga3.
namespace pagmo
{

namespace detail
{

// Solve the linear system Ax = b by Gaussian elimination with partial pivoting.
// Returns an empty optional when A is (numerically) singular.
PAGMO_DLL_PUBLIC std::optional<vector_double> gaussian_elimination(std::vector<std::vector<double>>,
                                                                   const vector_double &);

// Achievement Scalarization Function
PAGMO_DLL_PUBLIC double achievement(const vector_double &, const vector_double &);

// Perpendicular distance to reference point vectors
PAGMO_DLL_PUBLIC double perpendicular_distance(const std::vector<double> &, const std::vector<double> &);

/* Choose single random element from vector container.
 * The random engine is supplied by the caller so that the choice depends only
 * on the state of that engine, and not on the global pagmo::random_device.
 */
template <class T>
T choose_random_element(const std::vector<T> &container, random_engine_type &random_engine)
{
    std::uniform_int_distribution<typename std::vector<T>::size_type> dist(0u, container.size() - 1u);
    return container[dist(random_engine)];
}

/*  Adaptive normalisation of the objectives, Deb & Jain Algorithm 2.
 *
 *  Algorithm 2 operates on S_t, the union of the accepted fronts and the splitting
 *  front, and *not* on the whole combined population nor on the first front alone.
 *  Every function below therefore takes that set explicitly, as the vector
 *  `selected` of indices into the objective vectors of the combined population.
 *  Objectives, translated objectives and normalized objectives all stay indexed by
 *  the index an individual has in the combined population, so that the fronts and
 *  `selected` address them all in the same way.
 *
 *  Three coordinate systems appear here and are named consistently:
 *    - original:   the objective values as returned by the problem;
 *    - translated: original minus the ideal point, so that the ideal point is at
 *                  the origin and every member of S_t is in the non-negative orthant;
 *    - normalized: translated divided componentwise by the intercepts, so that the
 *                  hyperplane through the extreme points is the unit simplex.
 *
 *  The inter-generational memory is passed in explicitly rather than held by these
 *  functions: a null pointer means the corresponding quantity is not retained across
 *  generations, and a non-null one is read and then updated in place.
 */

// Ideal point of S_t; with memory, the best value for each objective since the start of the run
PAGMO_DLL_PUBLIC std::vector<double> nsga3_compute_ideal(const std::vector<vector_double> &objs,
                                                         const std::vector<pop_size_t> &selected,
                                                         std::vector<double> *running_ideal);

PAGMO_DLL_PUBLIC std::vector<std::vector<double>> nsga3_translate_objectives(const std::vector<vector_double> &objs,
                                                                             const std::vector<double> &ideal_point);

/*  Extreme point of S_t along each objective axis, Algorithm 2 line 4. Returned in
 *  the translated coordinates; retained extreme points are stored in the *original*
 *  coordinates, so that they stay meaningful as the ideal point moves.
 */
PAGMO_DLL_PUBLIC std::vector<std::vector<double>>
nsga3_find_extreme_points(const std::vector<pop_size_t> &selected,
                          const std::vector<std::vector<double>> &translated_objs,
                          const std::vector<double> &ideal_point, std::vector<std::vector<double>> *retained_extremes);

/*  Componentwise maximum of the translated objectives over S_t. This is the
 *  fallback NSGA-III prescribes whenever the hyperplane through the extreme points
 *  cannot be constructed, and it is deliberately *not* the nadir point: pagmo::nadir
 *  reports the worst values of the first non-dominated front only, whereas the
 *  fallback must see every member of S_t, dominated ones included.
 */
PAGMO_DLL_PUBLIC std::vector<double> nsga3_translated_maxima(const std::vector<std::vector<double>> &translated_objs,
                                                             const std::vector<pop_size_t> &selected);

PAGMO_DLL_PUBLIC std::vector<double> nsga3_find_intercepts(const std::vector<std::vector<double>> &ext_points,
                                                           const std::vector<std::vector<double>> &translated_objs,
                                                           const std::vector<pop_size_t> &selected);

PAGMO_DLL_PUBLIC std::vector<std::vector<double>>
nsga3_normalize_objectives(const std::vector<std::vector<double>> &translated_objs,
                           const std::vector<double> &intercepts);

/*  Environmental selection, Deb & Jain Algorithm 1.
 *
 *  Selects N_pop of the objective vectors of the combined parent and offspring
 *  populations for survival into the next generation, and returns their indices.
 *  `directions` is the immutable reference direction set built once per evolve()
 *  call; a working copy of it is reset and populated here. The memory pointers are
 *  those of nsga3_compute_ideal and nsga3_find_extreme_points.
 */
PAGMO_DLL_PUBLIC std::vector<pop_size_t> nsga3_selection(const std::vector<vector_double> &objs, pop_size_t N_pop,
                                                         const std::vector<reference_point> &directions,
                                                         std::vector<double> *running_ideal,
                                                         std::vector<std::vector<double>> *retained_extremes,
                                                         random_engine_type &reng);

} // namespace detail

} // namespace pagmo

#endif
