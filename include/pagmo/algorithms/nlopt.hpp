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

#include <pagmo/detail/nlopt_utils.hpp>
#include <pagmo/population.hpp>

namespace pagmo
{

class nlopt
{
    using nlopt_obj = detail::nlopt_obj;

public:
    nlopt() : nlopt(NLOPT_LN_COBYLA)
    {
    }
    explicit nlopt(::nlopt_algorithm algo) : m_algo(algo)
    {
    }
    population evolve(population pop) const
    {
        if (!pop.size()) {
            return pop;
        }
        auto &prob = pop.get_problem();
        nlopt_obj no(m_algo, prob);
        auto initial_guess = pop.get_x()[pop.best_idx()];
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
    ::nlopt_algorithm m_algo;
};
}

#endif
