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

#ifndef PAGMO_ALGORITHMS_DE_HPP
#define PAGMO_ALGORITHMS_DE_HPP

#include <algorithm>
#include <boost/any.hpp>
#include <boost/bimap.hpp>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include <pagmo/detail/nlopt_utils.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/type_traits.hpp>

namespace pagmo
{

namespace detail
{

template <typename = void>
struct nlopt_data {
    using names_map_t = boost::bimap<std::string, uncvref_t<decltype(NLOPT_LN_COBYLA)>>;
    static const names_map_t names;
};

inline typename nlopt_data<>::names_map_t nlopt_names_map()
{
    typename nlopt_data<>::names_map_t retval;
    using value_type = typename nlopt_data<>::names_map_t::value_type;
    retval.insert(value_type("cobyla", NLOPT_LN_COBYLA));
    retval.insert(value_type("bobyqa", NLOPT_LN_BOBYQA));
    retval.insert(value_type("praxis", NLOPT_LN_PRAXIS));
    retval.insert(value_type("neldermead", NLOPT_LN_NELDERMEAD));
    retval.insert(value_type("sbplx", NLOPT_LN_SBPLX));
    return retval;
}

template <typename T>
const typename nlopt_data<T>::names_map_t nlopt_data<T>::names = nlopt_names_map();
}

class nlopt
{
    using nlopt_obj = detail::nlopt_obj;
    using nlopt_data = detail::nlopt_data<>;

public:
    nlopt() : nlopt("neldermead")
    {
    }
    explicit nlopt(const std::string &algo)
        : m_algo(algo), m_select(std::string("best")), m_replace(std::string("best"))
    {
        if (nlopt_data::names.left.find(m_algo) == nlopt_data::names.left.end()) {
            std::ostringstream oss;
            std::transform(nlopt_data::names.left.begin(), nlopt_data::names.left.end(),
                           std::ostream_iterator<std::string>(oss, "\n"),
                           [](const uncvref_t<decltype(*nlopt_data::names.left.begin())> &v) { return v.first; });
            pagmo_throw(std::invalid_argument,
                        "unknown/unsupported NLopt algorithm '" + algo + "'. The valid algorithms are:\n" + oss.str());
        }
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
    population evolve(population pop) const
    {
        if (!pop.size()) {
            // In case of an empty pop, just return it.
            return pop;
        }

        auto &prob = pop.get_problem();
        nlopt_obj no(nlopt_data::names.left.at(m_algo), prob);

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
            }
        } else {
            assert(boost::any_cast<population::size_type>(&m_select));
        }

        if (initial_guess.size() != prob.get_nx()) {
            // TODO
            throw;
        }
        double fitness;
        const auto res = ::nlopt_optimize(no.m_value.get(), initial_guess.data(), &fitness);
        if (res < 0) {
            // TODO
            print(initial_guess, '\n');
            std::cout << "failed!!\n";
            throw;
        }
        print("Res: ", res, "\n");
        pop.set_xf(pop.best_idx(), initial_guess, {fitness});
        return pop;
    }

private:
    std::string m_algo;
    boost::any m_select;
    boost::any m_replace;
};
}

#endif
