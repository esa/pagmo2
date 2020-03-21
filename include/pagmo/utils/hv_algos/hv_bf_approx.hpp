/* Copyright 2017-2020 PaGMO development team

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

#ifndef PAGMO_UTILS_HV_BF_APPROX_HPP
#define PAGMO_UTILS_HV_BF_APPROX_HPP

#include <memory>
#include <string>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>

namespace pagmo
{

/// Bringmann-Friedrich approximation method
/**
 * This is the class containing the implementation of the Bringmann-Friedrich approximation method for the computation
 * of the least contributor to the hypervolume.
 * Default values for the parameters of the algorithm were obtained from the shark implementation of the
 * algorithm (http://image.diku.dk/shark/doxygen_pages/html/_least_contributor_approximator_8hpp_source.html)
 *
 * @see "Approximating the least hypervolume contributor: NP-hard in general, but fast in practice", Karl Bringmann,
 * Tobias Friedrich.
 *
 */
class PAGMO_DLL_PUBLIC bf_approx final : public hv_algorithm
{
public:
    /// Constructor
    /**
     * Constructs an instance of the algorithm
     *
     * @param use_exact boolean flag stating whether algorithm is allowed to use exact algorithms for the computation
     * @param trivial_subcase_size size of the sub-front (points overlapping the bounding box) for which algorithm
     * skips to the exact computation right away
     * @param eps accuracy of the approximation
     * @param delta confidence of the approximation
     * @param gamma constant used for computation of delta for each of the points during the sampling
     * @param delta_multiplier factor with which delta diminishes each round
     * @param initial_delta_coeff initial coefficient multiplied by the delta at round 0
     * @param alpha coefficicient stating how accurately current lowest contributor should be sampled
     * @param seed seeding for the pseudo-random number generator
     */
    bf_approx(bool use_exact = true, unsigned trivial_subcase_size = 1, double eps = 1e-2, double delta = 1e-6,
              double delta_multiplier = 0.775, double alpha = 0.2, double initial_delta_coeff = 0.1,
              double gamma = 0.25, unsigned seed = pagmo::random_device::next());

    // Compute hypervolume
    [[noreturn]] double compute(std::vector<vector_double> &, const vector_double &) const override;

    // Least contributor method
    unsigned long long least_contributor(std::vector<vector_double> &, const vector_double &) const override;

    // Greatest contributor method
    unsigned long long greatest_contributor(std::vector<vector_double> &, const vector_double &) const override;

    // Verify before compute method
    void verify_before_compute(const std::vector<vector_double> &, const vector_double &) const override;

    // Clone method.
    std::shared_ptr<hv_algorithm> clone() const override;

    // Algorithm name
    std::string get_name() const override;

private:
    // Compute delta for given point
    PAGMO_DLL_LOCAL double compute_point_delta(unsigned, vector_double::size_type, double) const;

    // Compute bounding box method
    PAGMO_DLL_LOCAL vector_double compute_bounding_box(const std::vector<vector_double> &, const vector_double &,
                                                       vector_double::size_type) const;

    // Determine whether point 'p' influences the volume of box (a, b)
    PAGMO_DLL_LOCAL int point_in_box(const vector_double &, const vector_double &, const vector_double &) const;

    // Performs a single round of sampling for given point at index 'idx'
    PAGMO_DLL_LOCAL void sampling_round(const std::vector<vector_double> &, double, unsigned, vector_double::size_type,
                                        double) const;

    // samples the bounding box and returns true if it fell into the exclusive hypervolume
    PAGMO_DLL_LOCAL bool sample_successful(const std::vector<vector_double> &, vector_double::size_type) const;

    enum extreme_contrib_type { LEAST = 1, GREATEST = 2 };

    // Approximated extreme contributor method
    PAGMO_DLL_LOCAL vector_double::size_type approx_extreme_contributor(
        std::vector<vector_double> &, const vector_double &, extreme_contrib_type, bool (*)(double, double),
        bool (*)(vector_double::size_type, vector_double::size_type, vector_double &, vector_double &),
        double (*)(vector_double::size_type, vector_double::size_type, vector_double &, vector_double &)) const;

    // ----------------
    PAGMO_DLL_LOCAL static double lc_end_condition(vector_double::size_type, vector_double::size_type, vector_double &,
                                                   vector_double &);

    PAGMO_DLL_LOCAL static double gc_end_condition(vector_double::size_type, vector_double::size_type, vector_double &,
                                                   vector_double &);

    PAGMO_DLL_LOCAL static bool lc_erase_condition(vector_double::size_type, vector_double::size_type, vector_double &,
                                                   vector_double &);

    PAGMO_DLL_LOCAL static bool gc_erase_condition(vector_double::size_type, vector_double::size_type, vector_double &,
                                                   vector_double &);

    // flag stating whether BF approximation should use exact computation for some exclusive hypervolumes
    const bool m_use_exact;

    // if the number of points overlapping the bounding box is small enough we can just compute that exactly
    // following variable states the number of points for which we perform the optimization
    const unsigned m_trivial_subcase_size;

    // accuracy of the approximation
    const double m_eps;

    // confidence of the approximation
    const double m_delta;

    // multiplier of the round delta value
    const double m_delta_multiplier;

    // alpha coefficient used for pushing on the sampling of the current least contributor
    const double m_alpha;

    // initial coefficient of the delta at round 0
    const double m_initial_delta_coeff;

    // constant used for the computation of point delta
    const double m_gamma;

    mutable detail::random_engine_type m_e;

    /**
     * 'least_contributor' method variables section
     *
     * Section below contains member variables that are relevant only to the least_contributor method.
     * They are not serialized as the members below are irrelevant outside of that scope.
     */
    // number of elementary operations performed for each point
    mutable std::vector<vector_double::size_type> m_no_ops;

    // number of samples for given box
    mutable std::vector<vector_double::size_type> m_no_samples;

    // number of "successful" samples that fell into the exclusive hypervolume
    mutable std::vector<vector_double::size_type> m_no_succ_samples;

    // stores the indices of points that were not yet removed during the progress of the algorithm
    mutable std::vector<vector_double::size_type> m_point_set;

    // exact hypervolumes of the bounding boxes
    mutable vector_double m_box_volume;

    // approximated exlusive hypervolume of each point
    mutable vector_double m_approx_volume;

    // deltas computed for each point using chernoff inequality component
    mutable vector_double m_point_delta;

    // pair (boxes[idx], points[idx]) form a box in which monte carlo sampling is performed
    mutable std::vector<vector_double> m_boxes;

    // list of indices of points that overlap the bounding box of each point
    // during monte carlo sampling it suffices to check only these points when deciding whether the sampling was
    // "successful"
    mutable std::vector<std::vector<vector_double::size_type>> m_box_points;
    /**
     * End of 'least_contributor' method variables section
     */
};
} // namespace pagmo

#endif
