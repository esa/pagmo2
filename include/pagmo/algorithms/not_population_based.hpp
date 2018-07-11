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

#include <boost/any.hpp>
#include <cassert>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/serialization.hpp>
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
class not_population_based
{
public:
    /// Default constructor.
    /**
     * The default constructor sets the selection and replacement policies to <tt>"best"</tt>,
     * and it initialises the RNG of the <tt>"random"</tt> selection/replacements policies via
     * random_device::next().
     */
    not_population_based()
        : m_select(std::string("best")), m_replace(std::string("best")), m_rselect_seed(random_device::next()),
          m_e(static_cast<std::mt19937::result_type>(m_rselect_seed))
    {
    }
    /// Set the seed for the ``"random"`` selection/replacement policies.
    /**
     * @param seed the value that will be used to seed the random number generator used by the ``"random"``
     * selection/replacement policies.
     */
    void set_random_sr_seed(unsigned seed)
    {
        m_rselect_seed = seed;
        m_e.seed(static_cast<std::mt19937::result_type>(m_rselect_seed));
    }
    /// Set the individual selection policy.
    /**
     * This method will set the policy that is used to select the individual
     * that will be optimised when calling the ``evolve()`` method of the algorithm.
     *
     * The input string must be one of ``"best"``, ``"worst"`` and ``"random"``:
     * - ``"best"`` will select the best individual in the population,
     * - ``"worst"`` will select the worst individual in the population,
     * - ``"random"`` will randomly choose one individual in the population.
     *
     * set_random_sr_seed() can be used to seed the random number generator used by the ``"random"`` policy.
     *
     * Instead of a selection policy, a specific individual in the population can be selected via
     * set_selection(population::size_type).
     *
     * @param select the selection policy.
     *
     * @throws std::invalid_argument if \p select is not one of ``"best"``, ``"worst"`` or ``"random"``.
     */
    void set_selection(const std::string &select)
    {
        if (select != "best" && select != "worst" && select != "random") {
            pagmo_throw(std::invalid_argument,
                        "the individual selection policy must be one of ['best', 'worst', 'random'], but '" + select
                            + "' was provided instead");
        }
        m_select = select;
    }
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
    /// Get the individual selection policy or index.
    /**
     * This method will return a \p boost::any containing either the individual selection policy (as an \p std::string)
     * or the individual selection index (as a population::size_type). The selection policy or index is set via
     * set_selection(const std::string &) and set_selection(population::size_type).
     *
     * @return the individual selection policy or index.
     */
    boost::any get_selection() const
    {
        return m_select;
    }
    /// Set the individual replacement policy.
    /**
     * This method will set the policy that is used in the ``evolve()`` method of the algorithm to select the individual
     * that will be replaced by the optimised individual.
     *
     * The input string must be one of ``"best"``, ``"worst"`` and ``"random"``:
     * - ``"best"`` will select the best individual in the population,
     * - ``"worst"`` will select the worst individual in the population,
     * - ``"random"`` will randomly choose one individual in the population.
     *
     * set_random_sr_seed() can be used to seed the random number generator used by the ``"random"`` policy.
     *
     * Instead of a replacement policy, a specific individual in the population can be selected via
     * set_replacement(population::size_type).
     *
     * @param replace the replacement policy.
     *
     * @throws std::invalid_argument if \p replace is not one of ``"best"``, ``"worst"`` or ``"random"``.
     */
    void set_replacement(const std::string &replace)
    {
        if (replace != "best" && replace != "worst" && replace != "random") {
            pagmo_throw(std::invalid_argument,
                        "the individual replacement policy must be one of ['best', 'worst', 'random'], but '" + replace
                            + "' was provided instead");
        }
        m_replace = replace;
    }
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
    /// Get the individual replacement policy or index.
    /**
     * This method will return a \p boost::any containing either the individual replacement policy (as an \p
     * std::string) or the individual replacement index (as a population::size_type). The replacement policy or index is
     * set via set_replacement(const std::string &) and set_replacement(population::size_type).
     *
     * @return the individual replacement policy or index.
     */
    boost::any get_replacement() const
    {
        return m_replace;
    }
    /// Save to archive.
    /**
     * @param ar the target archive.
     *
     * @throws unspecified any exception thrown by the serialization of primitive types.
     */
    template <typename Archive>
    void save(Archive &ar) const
    {
        if (boost::any_cast<std::string>(&m_select)) {
            // NOTE: true -> string, false -> idx.
            ar(true);
            ar(boost::any_cast<std::string>(m_select));
        } else {
            ar(false);
            ar(boost::any_cast<population::size_type>(m_select));
        }
        if (boost::any_cast<std::string>(&m_replace)) {
            // NOTE: true -> string, false -> idx.
            ar(true);
            ar(boost::any_cast<std::string>(m_replace));
        } else {
            ar(false);
            ar(boost::any_cast<population::size_type>(m_replace));
        }
        ar(m_rselect_seed, m_e);
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
    void load(Archive &ar)
    {
        not_population_based tmp;
        bool flag;
        std::string str;
        population::size_type idx;
        ar(flag);
        if (flag) {
            ar(str);
            tmp.m_select = str;
        } else {
            ar(idx);
            tmp.m_select = idx;
        }
        ar(flag);
        if (flag) {
            ar(str);
            tmp.m_replace = str;
        } else {
            ar(idx);
            tmp.m_replace = idx;
        }
        ar(tmp.m_rselect_seed, tmp.m_e);
        *this = std::move(tmp);
    }

protected:
    /// Select individual.
    /**
     * This method will select a single individual from the input population \p pop, returning
     * its decision vector and fitness as a pair. The selection is done according to the currently
     * active selection policy:
     * - if not_population_based::m_select is <tt>"best"</tt>, then the decision and fitness vectors of the
     *   best individual are returned,
     * - if not_population_based::m_select is <tt>"worst"</tt>, then the decision and fitness vectors of the
     *   worst individual are returned,
     * - if not_population_based::m_select is <tt>"random"</tt>, then the decision and fitness vectors of a
     *   randomly-selected individual are returned,
     * - if not_population_based::m_select is an index, then the decision and fitness vectors of the individual
     *   at the corresponding position in the population are returned.
     *
     * Note that selecting a best or worst individual is meaningful only in single-objective
     * problems.
     *
     * @param pop the input population.
     *
     * @return a pair containing the decision and fitness vectors of the selected individual.
     *
     * @throws std::invalid_argument if not_population_based::m_select is an index and the index is not smaller than the
     * size of \p pop.
     * @throws unspecified any exception thrown by population::best_idx() or population::worst_idx().
     */
    std::pair<vector_double, vector_double> select_individual(const population &pop) const
    {
        vector_double x, f;
        if (boost::any_cast<std::string>(&m_select)) {
            const auto &s_select = boost::any_cast<const std::string &>(m_select);
            if (s_select == "best") {
                x = pop.get_x()[pop.best_idx()];
                f = pop.get_f()[pop.best_idx()];
            } else if (s_select == "worst") {
                x = pop.get_x()[pop.worst_idx()];
                f = pop.get_f()[pop.worst_idx()];
            } else {
                assert(s_select == "random");
                std::uniform_int_distribution<population::size_type> dist(0, pop.size() - 1u);
                const auto idx = dist(m_e);
                x = pop.get_x()[idx];
                f = pop.get_f()[idx];
            }
        } else {
            const auto idx = boost::any_cast<population::size_type>(m_select);
            if (idx >= pop.size()) {
                pagmo_throw(std::invalid_argument, "cannot select the individual at index " + std::to_string(idx)
                                                       + ": the population has a size of only "
                                                       + std::to_string(pop.size()));
            }
            x = pop.get_x()[idx];
            f = pop.get_f()[idx];
        }
        return std::make_pair(std::move(x), std::move(f));
    }
    /// Replace individual.
    /**
     * This method will replace a single individual in the input population \p pop, setting its decision
     * vector to \p x and its fitness vector to \p f. The selection is done according to the currently
     * active selection policy:
     * - if not_population_based::m_replace is <tt>"best"</tt>, then the best individual will be replaced,
     * - if not_population_based::m_replace is <tt>"worst"</tt>, then the worst individual will be replaced,
     * - if not_population_based::m_replace is <tt>"random"</tt>, then a randomly-selected individual will be replaced,
     * - if not_population_based::m_replace is an index, then the individual at the corresponding position in the
     *   population will be replaced.
     *
     * Note that selecting a best or worst individual is meaningful only in single-objective
     * problems.
     *
     * @param pop the input population.
     * @param x the decision vector of the new individual.
     * @param f the fitness vector of the new individual.
     *
     * @throws std::invalid_argument if not_population_based::m_replace is an index and the index is not smaller than
     * the size of \p pop.
     * @throws unspecified any exception thrown by population::best_idx(), population::worst_idx(),
     * or population::set_xf().
     */
    void replace_individual(population &pop, const vector_double &x, const vector_double &f) const
    {
        if (boost::any_cast<std::string>(&m_replace)) {
            const auto &s_replace = boost::any_cast<const std::string &>(m_replace);
            if (s_replace == "best") {
                pop.set_xf(pop.best_idx(), x, f);
            } else if (s_replace == "worst") {
                pop.set_xf(pop.worst_idx(), x, f);
            } else {
                assert(s_replace == "random");
                std::uniform_int_distribution<population::size_type> dist(0, pop.size() - 1u);
                pop.set_xf(dist(m_e), x, f);
            }
        } else {
            const auto idx = boost::any_cast<population::size_type>(m_replace);
            if (idx >= pop.size()) {
                pagmo_throw(std::invalid_argument, "cannot replace the individual at index " + std::to_string(idx)
                                                       + ": the population has a size of only "
                                                       + std::to_string(pop.size()));
            }
            pop.set_xf(idx, x, f);
        }
    }

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

#endif
