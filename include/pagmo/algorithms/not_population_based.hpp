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

#ifndef PAGMO_NOT_POPULATION_BASED_HPP
#define PAGMO_NOT_POPULATION_BASED_HPP

#include <string>
#include <utility>

#include <boost/any.hpp>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// Base class for non population based solvers
/**
 * This class contains common methods useful in the implementation of user-defined algorithms
 * that are not population based and establishes a common interface to a population
 *
 * Currently, this class implements policies for the selection and replacement of a single individual
 * in a population, which are meant to be used in the implementation of the <tt>evolve()</tt> method of the
 * user-defined algorithm (see, e.g., pagmo::nlopt::evolve()).
 */
class PAGMO_DLL_PUBLIC not_population_based
{
public:
    // Default constructor.
    not_population_based();
    // Set the seed for the ``"random"`` selection/replacement policies.
    void set_random_sr_seed(unsigned);
    // Set the individual selection policy.
    void set_selection(const std::string &);
    /// Set the individual selection index.
    /**
     * This method will set the index of the individual that is selected for optimisation
     * in the ``evolve()`` method of the algorithm.
     *
     * @param n the index in the population of the individual to be selected for optimisation.
     */
    void set_selection(population::size_type n)
    {
        m_select = n;
    }
    // Get the individual selection policy or index.
    boost::any get_selection() const;
    // Set the individual replacement policy.
    void set_replacement(const std::string &);
    /// Set the individual replacement index.
    /**
     * This method will set the index of the individual that is replaced after the optimisation
     * in the ``evolve()`` method of the algorithm.
     *
     * @param n the index in the population of the individual to be replaced after the optimisation.
     */
    void set_replacement(population::size_type n)
    {
        m_replace = n;
    }
    // Get the individual replacement policy or index.
    boost::any get_replacement() const;
    /// Save to archive.
    /**
     * @param ar the target archive.
     *
     * @throws unspecified any exception thrown by the serialization of primitive types.
     */
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        if (boost::any_cast<std::string>(&m_select)) {
            // NOTE: true -> string, false -> idx.
            ar << true;
            ar << boost::any_cast<std::string>(m_select);
        } else {
            ar << false;
            ar << boost::any_cast<population::size_type>(m_select);
        }
        if (boost::any_cast<std::string>(&m_replace)) {
            // NOTE: true -> string, false -> idx.
            ar << true;
            ar << boost::any_cast<std::string>(m_replace);
        } else {
            ar << false;
            ar << boost::any_cast<population::size_type>(m_replace);
        }
        ar << m_rselect_seed;
        ar << m_e;
    }
    /// Load from archive.
    /**
     * In case of exceptions, \p this will be unaffected.
     *
     * @param ar the source archive.
     *
     * @throws unspecified any exception thrown by the deserialization of primitive types.
     */
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        not_population_based tmp;
        bool flag;
        std::string str;
        population::size_type idx;
        ar >> flag;
        if (flag) {
            ar >> str;
            tmp.m_select = str;
        } else {
            ar >> idx;
            tmp.m_select = idx;
        }
        ar >> flag;
        if (flag) {
            ar >> str;
            tmp.m_replace = str;
        } else {
            ar >> idx;
            tmp.m_replace = idx;
        }
        ar >> tmp.m_rselect_seed;
        ar >> tmp.m_e;
        *this = std::move(tmp);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

protected:
    // Select individual.
    std::pair<vector_double, vector_double> select_individual(const population &) const;
    // Replace individual.
    void replace_individual(population &, const vector_double &, const vector_double &) const;

protected:
    /// Individual selection policy.
    /**
     * This \p boost::any instance must contain either an \p std::string or a population::size_type,
     * otherwise the behaviour will be undefined.
     */
    boost::any m_select;
    /// Individual replacement policy.
    /**
     * This \p boost::any instance must contain either an \p std::string or a population::size_type,
     * otherwise the behaviour will be undefined.
     */
    boost::any m_replace;
    /// Seed for the <tt>"random"</tt> selection/replacement policies.
    unsigned m_rselect_seed;
    /// Random engine for the <tt>"random"</tt> selection/replacement policies.
    mutable detail::random_engine_type m_e;
};

} // namespace pagmo

// Disable tracking for the serialisation of not_population_based.
BOOST_CLASS_TRACKING(pagmo::not_population_based, boost::serialization::track_never)

#endif
