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

#ifndef PAGMO_PROBLEMS_INVENTORY_HPP
#define PAGMO_PROBLEMS_INVENTORY_HPP

#include <string>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// Stochastic Programming Test Problem: An Inventory Model
/**
 *
 * \image html newsvendor.png "A news vendor icon" width=3cm
 *
 * This problem is a generalization of the simple inventory problem so-called of the "news-vendor",
 * widely used to introduce the main tools and techniques of stochastic programming
 * Assume you are a newsvendor and each week, for the next \f$ N\f$ weeks, you need to decide how many
 * journals to order (indicated with the decision variable  \f$ x_i \f$). The weekly journal demand is
 * unknown to you and is indicated with the variable \f$d_i\f$. The cost of
 * ordering journals before the week starts is \f$ c\f$, the cost of ordering journals during the week
 * (in order to meet an unforeseen demand) is \f$ b \f$ and the cost of having to hold unsold journals
 * is \f$ h \f$. The inventory level of journals will be defined by the succession:
 * \f[
 *  I_i = [I_{i-1} + x_i - d_i]_+, I_1 = 0
 * \f]
 * while the total cost of running the journal sales for \f$N\f$ weeks will be:
 * \f[
 *  J(\mathbf x, \mathbf d) = c \sum_{i=1}^N x_i+ b \sum_{i=1}^N [d_i - I_i - x_i]_+ + h \sum_{i=1}^N [I_i + x_i -
 * d_i]_+
 * \f]
 *
 * See: www2.isye.gatech.edu/people/faculty/Alex_Shapiro/SPbook.pdf
 *
 */
class PAGMO_DLL_PUBLIC inventory
{
public:
    /// Constructor from weeks, sample size and random seed
    /**
     * Given the numer of weeks (i.e. prolem dimension), the sample size to
     * approximate the expected value and a starting random seed, we construct
     * the inventory prolem
     *
     * @param weeks dimension of the problem corresponding to the numer of weeks
     * to plan the inventory for.
     * @param sample_size dimension of the sample used to approximate the expected value
     * @param seed starting random seed to build the pseudorandom sequences used to
     * generate the sample
     */
    inventory(unsigned weeks = 4u, unsigned sample_size = 10u, unsigned seed = pagmo::random_device::next())
        : m_weeks(weeks), m_sample_size(sample_size), m_e(seed), m_seed(seed)
    {
    }
    // Fitness computation
    vector_double fitness(const vector_double &) const;
    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;
    /// Sets the seed
    /**
     * @param seed the random number generator seed
     */
    void set_seed(unsigned seed)
    {
        m_seed = seed;
    }
    /// Problem name
    /**
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        return "Inventory problem";
    }
    // Extra info
    std::string get_extra_info() const;

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    // Number of weeks to plan for
    unsigned m_weeks;
    // Sample size
    unsigned m_sample_size;
    // Random engine
    mutable detail::random_engine_type m_e;
    // Seed
    unsigned m_seed;
};

} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::inventory)

#endif
