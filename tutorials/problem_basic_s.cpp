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

// In this tutorial, we will learn how to code a
// stochastic problem. We will focus on the trivial minimization
// problem where the objective is (as in the previous tutorials)
// f = x1^2 + x2^2 + x3^2 + x4^2 + s
// in the bounds: -10 <= xi <= 10. Note that s is our stochastic
// variable which we consider normally distributed around 0 with standard
// deviation equal to 1.

// All we need to do is to implement a class having the
// following mandatory methods:
//
// fitness_vector fitness(const decision_vector &) const
// std::pair<decision_vector, decision_vector> get_bounds() const
//
// and add the method
//
// void set_seed(unsigned)
//
// which will be used to change the seed of the random engine

#include <string>

#include <pagmo/io.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;
class problem_basic_s
{
public:
    // Since the problem is stochastic we introduce as data members a random engine
    // and a seed
    problem_basic_s(unsigned seed = pagmo::random_device::next()) : m_e(seed), m_seed(seed) {}

    // Mandatory, computes ... well ... the fitness.
    // In a stochastic problem the fitness depends on the
    // chromosome (decision vector) but also on a number
    // of stochastic variables which are instantiated from a
    // common seed s
    vector_double fitness(const vector_double &x) const
    {
        // We seed the random engine
        m_e.seed(m_seed);
        // We define a normal distribution
        auto s = std::normal_distribution<double>(0., 1.);
        // We return the fitness
        return {x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] + s(m_e)};
    }

    // Mandatory, returns the box-bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{-10, -10, -10, -10}, {10, 10, 10, 10}};
    }

    // Mandatory for the problem to be stochastic
    void set_seed(unsigned seed)
    {
        m_seed = seed;
    }

    // Optional, provides a name for the problem overrding the default name
    std::string get_name() const
    {
        return "My Problem";
    }

    // Optional, provides extra information that will be appended after
    // the default stream operator
    std::string get_extra_info() const
    {
        return "This is just a simple toy problem with one objective,\n no constraints and a fixed dimension of 4.\n";
    }

    // Optional methods-data can also be accessed later via
    // the problem::extract() method
    vector_double best_known() const
    {
        return {0., 0., 0., 0.};
    }

private:
    // Random engine
    mutable detail::random_engine_type m_e;
    // Seed
    unsigned m_seed;
};

int main()
{
    // Constructing a problem
    problem p0{problem_basic_s{}};
    // Streaming to screen the problem
    std::cout << p0 << '\n';
    // Getting its dimensions
    std::cout << "Calling the dimension getter: " << p0.get_nx() << '\n';
    std::cout << "Calling the fitness dimension getter: " << p0.get_nobj() << '\n';

    // Getting the bounds via the pagmo::print eating also std containers
    pagmo::print("Calling the bounds getter: ", p0.get_bounds(), "\n");

    // As soon as a problem its created its function evaluation counter
    // is set to zero. Checking its value is easy
    pagmo::print("fevals: ", p0.get_fevals(), "\n");
    // Computing one fitness
    pagmo::print("calling fitness in x=[2,2,2,2]: ", p0.fitness({2, 2, 2, 2}), "\n");
    // The evaluation counter is now ... well ... 1
    pagmo::print("fevals: ", p0.get_fevals(), "\n");
    // Changing the seed
    p0.set_seed(3245323u);
    pagmo::print("seed changed to: " + std::to_string(3245323u), '\n');
    // Recomputing the fitness which will be ... well ... different
    pagmo::print("calling fitness in x=[2,2,2,2]: ", p0.fitness({2, 2, 2, 2}), "\n");

    // While our problem_basic_s struct is now hidden inside the pagmo::problem
    // we can still access its methods / data via the extract interface
    pagmo::print("Accessing best_known: ", p0.extract<problem_basic_s>()->best_known(), "\n");
}
