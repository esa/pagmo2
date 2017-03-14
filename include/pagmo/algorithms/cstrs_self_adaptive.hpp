/* Copyright 2017 PaGMO development team

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

#ifndef PAGMO_ALGORITHMS_CSTRS_SELF_ADAPTIVE_HPP
#define PAGMO_ALGORITHMS_CSTRS_SELF_ADAPTIVE_HPP

#include <iomanip>
#include <random>
#include <string>
#include <tuple>

#include "../algorithm.hpp"
#include "../exceptions.hpp"
#include "../io.hpp"
#include "../population.hpp"
#include "../rng.hpp"
#include "../utils/generic.hpp"

namespace pagmo
{
namespace detail
{
/// Constrainted self adaptive udp
/**
 * Implements a udp that wraps a population and results in self adaptive constraints handling.
 *
 * The key idea of this constraint handling technique is to represent the constraint violation by one single
 * infeasibility measure, and to adapt dynamically the penalization of infeasible solutions. As the penalization process
 * depends on a given population, a method to update the penalties to a new population is provided.
 *
 * @see Farmani R., & Wright, J. A. (2003). Self-adaptive fitness formulation for constrained optimization.
 * Evolutionary Computation, IEEE Transactions on, 7(5), 445-455 for the paper introducing the method.
 *
 */
class unconstrain_with_adaptive_penalty
{
public:
    /// Constructs the udp. At construction all
    unconstrain_with_adaptive_penalty(const population &pop)
    {
    }

private:
    // According to the population the first penalty may or may not be applied
    bool m_apply_penalty_1;
    double m_scaling_factor;

    constraint_vector m_c_scaling;

    fitness_vector m_f_hat_down;
    fitness_vector m_f_hat_up;
    fitness_vector m_f_hat_round;

    double m_i_hat_down;
    double m_i_hat_up;
    double m_i_hat_round;
}
}

} // namespace pagmo

// PAGMO_REGISTER_ALGORITHM(pagmo::sea)

#endif
