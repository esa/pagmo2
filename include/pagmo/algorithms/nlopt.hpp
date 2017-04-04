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

#ifndef PAGMO_ALGORITHMS_NLOPT_HPP
#define PAGMO_ALGORITHMS_NLOPT_HPP

#include <algorithm>
#include <boost/any.hpp>
#include <cassert>
#include <cmath>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <pagmo/detail/nlopt_utils.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/type_traits.hpp>

namespace pagmo
{

// TODO
// - cache
// - optimisation for dense gradients
// - error messages mentioning some algos don't support constraints etc.
class nlopt
{
    using nlopt_obj = detail::nlopt_obj;
    using nlopt_data = detail::nlopt_data<>;

public:
    using log_line_type = std::tuple<unsigned long, double, vector_double>;
    using log_type = std::vector<log_line_type>;

private:
    static_assert(std::is_same<log_line_type, detail::nlopt_obj::log_line_type>::value, "Invalid log line type.");

public:
    nlopt() : nlopt("sbplx")
    {
    }
    explicit nlopt(const std::string &algo)
        : m_algo(algo), m_select(std::string("best")), m_replace(std::string("best")),
          m_rselect_seed(random_device::next()), m_e(static_cast<std::mt19937::result_type>(m_rselect_seed))
    {
        // Check version.
        int major, minor, bugfix;
        ::nlopt_version(&major, &minor, &bugfix);
        if (major < 2) {
            pagmo_throw(std::runtime_error, "Only NLopt version >= 2 is supported");
        }

        // Check the algorithm.
        if (nlopt_data::names.left.find(m_algo) == nlopt_data::names.left.end()) {
            // The selected algorithm is unknown or not among the supported ones.
            std::ostringstream oss;
            std::transform(nlopt_data::names.left.begin(), nlopt_data::names.left.end(),
                           std::ostream_iterator<std::string>(oss, "\n"),
                           [](const uncvref_t<decltype(*nlopt_data::names.left.begin())> &v) { return v.first; });
            pagmo_throw(std::invalid_argument, "unknown/unsupported NLopt algorithm '" + algo
                                                   + "'. The supported algorithms are:\n" + oss.str());
        }
    }
    void set_random_selection_seed(unsigned seed)
    {
        m_rselect_seed = seed;
        m_e.seed(static_cast<std::mt19937::result_type>(m_rselect_seed));
    }
    void set_selection(const std::string &select)
    {
        if (select != "best" && select != "worst" && select != "random") {
            pagmo_throw(std::invalid_argument,
                        "the individual selection policy must be one of ['best', 'worst', 'random'], but '" + select
                            + "' was provided instead");
        }
        m_select = select;
    }
    void set_selection(population::size_type n)
    {
        m_select = n;
    }
    boost::any get_selection() const
    {
        return m_select;
    }
    void set_replacement(const std::string &replace)
    {
        if (replace != "best" && replace != "worst" && replace != "random") {
            pagmo_throw(std::invalid_argument,
                        "the individual replacement policy must be one of ['best', 'worst', 'random'], but '" + replace
                            + "' was provided instead");
        }
        m_replace = replace;
    }
    void set_replacement(population::size_type n)
    {
        m_replace = n;
    }
    boost::any get_replacement() const
    {
        return m_replace;
    }
    population evolve(population pop) const
    {
        if (!pop.size()) {
            // In case of an empty pop, just return it.
            return pop;
        }

        auto &prob = pop.get_problem();
        const auto nc = prob.get_nc();

        // Create the nlopt obj.
        // NOTE: this will check also the problem's properties.
        nlopt_obj no(nlopt_data::names.left.at(m_algo), prob, m_sc_stopval, m_sc_ftol_rel, m_sc_ftol_abs, m_sc_xtol_rel,
                     m_sc_xtol_abs, m_sc_maxeval, m_sc_maxtime, m_verbosity);

        // Setup of the initial guess.
        vector_double initial_guess;
        if (boost::any_cast<std::string>(&m_select)) {
            const auto &s_select = boost::any_cast<const std::string &>(m_select);
            if (s_select == "best") {
                initial_guess = pop.get_x()[pop.best_idx()];
            } else if (s_select == "worst") {
                initial_guess = pop.get_x()[pop.worst_idx()];
            } else {
                assert(s_select == "random");
                std::uniform_int_distribution<population::size_type> dist(0, pop.size() - 1u);
                initial_guess = pop.get_x()[dist(m_e)];
            }
        } else {
            const auto idx = boost::any_cast<population::size_type>(m_select);
            if (idx >= pop.size()) {
                pagmo_throw(std::out_of_range, "cannot select the individual at index " + std::to_string(idx)
                                                   + " for evolution: the population has a size of only "
                                                   + std::to_string(pop.size()));
            }
            initial_guess = pop.get_x()[idx];
        }
        // Check the initial guess.
        // NOTE: this should be guaranteed by the population's invariants.
        assert(initial_guess.size() == prob.get_nx());
        const auto bounds = prob.get_bounds();
        for (decltype(bounds.first.size()) i = 0; i < bounds.first.size(); ++i) {
            if (std::isnan(initial_guess[i])) {
                pagmo_throw(std::invalid_argument,
                            "the value of the initial guess at index " + std::to_string(i) + " is NaN");
            }
            if (initial_guess[i] < bounds.first[i] || initial_guess[i] > bounds.second[i]) {
                pagmo_throw(std::invalid_argument, "the value of the initial guess at index " + std::to_string(i)
                                                       + " is outside the problem's bounds");
            }
        }

        // Run the optimisation and store the status returned by NLopt.
        double fitness;
        m_last_opt_result = ::nlopt_optimize(no.m_value.get(), initial_guess.data(), &fitness);
        if (m_verbosity) {
            // Print to screen the result of the optimisation, if we are being verbose.
            std::cout << "\nOptimisation return status: " << detail::nlopt_res2string(m_last_opt_result) << '\n';
        }

        // Replace the log.
        m_log = std::move(no.m_log);

        // Store the new individual into the population.
        if (boost::any_cast<std::string>(&m_replace)) {
            const auto &s_replace = boost::any_cast<const std::string &>(m_replace);
            if (s_replace == "best") {
                if (nc) {
                    pop.set_x(pop.best_idx(), initial_guess);
                } else {
                    pop.set_xf(pop.best_idx(), initial_guess, {fitness});
                }
            } else if (s_replace == "worst") {
                if (nc) {
                    pop.set_x(pop.worst_idx(), initial_guess);
                } else {
                    pop.set_xf(pop.worst_idx(), initial_guess, {fitness});
                }
            } else {
                assert(s_replace == "random");
                std::uniform_int_distribution<population::size_type> dist(0, pop.size() - 1u);
                if (nc) {
                    pop.set_x(dist(m_e), initial_guess);
                } else {
                    pop.set_xf(dist(m_e), initial_guess, {fitness});
                }
            }
        } else {
            const auto idx = boost::any_cast<population::size_type>(m_replace);
            if (idx >= pop.size()) {
                pagmo_throw(std::out_of_range, "cannot replace the individual at index " + std::to_string(idx)
                                                   + " after evolution: the population has a size of only "
                                                   + std::to_string(pop.size()));
            }
            if (nc) {
                pop.set_x(idx, initial_guess);
            } else {
                pop.set_xf(idx, initial_guess, {fitness});
            }
        }

        // Return the evolved pop.
        return pop;
    }
    std::string get_name() const
    {
        return "NLopt - " + m_algo;
    }
    void set_verbosity(unsigned n)
    {
        m_verbosity = n;
    }
    std::string get_extra_info() const
    {
        int major, minor, bugfix;
        ::nlopt_version(&major, &minor, &bugfix);
        return "\tNLopt version: " + std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(bugfix)
               + "\n\tLast optimisation return code: " + detail::nlopt_res2string(m_last_opt_result) + "\n\tVerbosity: "
               + std::to_string(m_verbosity) + "\n\tIndividual selection "
               + (boost::any_cast<population::size_type>(&m_select)
                      ? "idx: " + std::to_string(boost::any_cast<population::size_type>(m_select))
                      : "policy: " + boost::any_cast<std::string>(m_select))
               + "\n\tIndividual replacement "
               + (boost::any_cast<population::size_type>(&m_replace)
                      ? "idx: " + std::to_string(boost::any_cast<population::size_type>(m_replace))
                      : "policy: " + boost::any_cast<std::string>(m_replace))
               + "\n\tStopping criteria:\n\t\tstopval:  "
               + (m_sc_stopval == -HUGE_VAL ? "disabled" : detail::to_string(m_sc_stopval)) + "\n\t\tftol_rel: "
               + (m_sc_ftol_rel <= 0. ? "disabled" : detail::to_string(m_sc_ftol_rel)) + "\n\t\tftol_abs: "
               + (m_sc_ftol_abs <= 0. ? "disabled" : detail::to_string(m_sc_ftol_abs)) + "\n\t\txtol_rel: "
               + (m_sc_xtol_rel <= 0. ? "disabled" : detail::to_string(m_sc_xtol_rel)) + "\n\t\txtol_abs: "
               + (m_sc_xtol_abs <= 0. ? "disabled" : detail::to_string(m_sc_xtol_abs)) + "\n\t\tmaxeval:  "
               + (m_sc_maxeval <= 0. ? "disabled" : detail::to_string(m_sc_maxeval)) + "\n\t\tmaxtime:  "
               + (m_sc_maxtime <= 0. ? "disabled" : detail::to_string(m_sc_maxtime)) + "\n";
    }

private:
    std::string m_algo;
    boost::any m_select;
    boost::any m_replace;
    unsigned m_rselect_seed;
    mutable detail::random_engine_type m_e;
    mutable ::nlopt_result m_last_opt_result = NLOPT_SUCCESS;
    // Stopping criteria.
    double m_sc_stopval = -HUGE_VAL;
    double m_sc_ftol_rel = 0.;
    double m_sc_ftol_abs = 0.;
    double m_sc_xtol_rel = 1E-8;
    double m_sc_xtol_abs = 0.;
    int m_sc_maxeval = 0;
    int m_sc_maxtime = 0;
    // Verbosity/log.
    unsigned m_verbosity = 0;
    mutable log_type m_log;
};
}

#endif
