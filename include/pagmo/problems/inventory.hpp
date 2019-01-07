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

#ifndef PAGMO_PROBLEM_INVENTORY_HPP
#define PAGMO_PROBLEM_INVENTORY_HPP

#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
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
class inventory
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
    inventory(unsigned int weeks = 4u, unsigned int sample_size = 10u, unsigned int seed = pagmo::random_device::next())
        : m_weeks(weeks), m_sample_size(sample_size), m_e(seed), m_seed(seed)
    {
    }
    /// Fitness computation
    /**
     * Computes the fitness for this UDP
     *
     * @param x the decision vector.
     *
     * @return the fitness of \p x.
     */
    vector_double fitness(const vector_double &x) const
    {
        // We seed the random engine
        m_e.seed(m_seed);
        // We construct a uniform distribution from 0 to 1.
        auto drng = std::uniform_real_distribution<double>(0., 1.);
        // We may now start the computations
        const double c = 1.0, b = 1.5,
                     h = 0.1; // c is the cost per unit, b is the backorder penalty cost and h is the holding cost
        double retval = 0;

        for (decltype(m_sample_size) i = 0; i < m_sample_size; ++i) {
            double I = 0;
            for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
                double d = drng(m_e) * 100;
                retval += c * x[j] + b * std::max<double>(d - I - x[j], 0) + h * std::max<double>(I + x[j] - d, 0);
                I = std::max<double>(0, I + x[j] - d);
            }
        }
        return {retval / m_sample_size};
    }
    /// Box-bounds
    /**
     *
     * It returns the box-bounds for this UDP.
     *
     * @return the lower and upper bounds for each of the decision vector components
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        vector_double lb(m_weeks, 0.);
        vector_double ub(m_weeks, 200.);
        return {lb, ub};
    }
    /// Sets the seed
    /**
     *
     *
     * @param seed the random number generator seed
     */
    void set_seed(unsigned int seed)
    {
        m_seed = seed;
    }
    /// Problem name
    /**
     *
     *
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        return "Inventory problem";
    }
    /// Extra informations
    /**
     *
     *
     * @return a string containing extra informations on the problem
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        ss << "\tWeeks: " << std::to_string(m_weeks) << "\n";
        ss << "\tSample size: " << std::to_string(m_sample_size) << "\n";
        ss << "\tSeed: " << std::to_string(m_seed) << "\n";
        return ss.str();
    }

    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_weeks, m_sample_size, m_e, m_seed);
    }

private:
    // Number of weeks to plan for
    unsigned int m_weeks;
    // Sample size
    unsigned int m_sample_size;
    // Random engine
    mutable detail::random_engine_type m_e;
    // Seed
    unsigned int m_seed;
};

} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::inventory)

#endif
