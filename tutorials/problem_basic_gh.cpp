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

// In this tutorial we learn how to implement the gradients and the hessians
// of each of fitness function components.
//
// Consider the simple problem:
// f = x1^2 + x2^2 + x3^2 + x4^2 in the bounds:
// -10 <= xi <= 10
//

// All we need to do is to implement a struct (or class) having the
// following mandatory methods:
//
// vector_double fitness(const vector_double &) const
// std::pair<vector_double, vector_double> get_bounds() const
//
// And add the methods:
// vector_double gradient(const vector_double &x) const
// std::vector<vector_double> hessians(const vector_double &) const
//
// Gradient:
// In PaGMO, each component fi of the fitness may be associated
// with its gradient dfi/dxj. The user may implement the optional
// method:
//
// vector_double gradient(const vector_double &x) const
//
// returning a vector_double containing dfi/dxj in the order defined by the
// gradient sparsity pattern which, by default is dense and is:
// [(0,0),(0,1), .., (1,0), (1,1), ..]
//
// The default dense gradient sparsity can be overridden by
// adding one optional method
//
// sparsity_pattern gradient_sparsity() const
//
// The gradient sparsity pattern is a std::vector of pairs (i,j)
// containing the indeces of non null entries of the gradients.
// Note that the dimensions of the sparsity pattern of the gradients
// must match that of the value returned by the implemented gradient
// method
//
//
// Hessians:
// In PaGMO each component fk of the fitness
// may be associated to an Hessian containing d^2fk/dxj/dxi.
// The user may implement the additional method:
//
// std::vector<vector_double> hessians(const vector_double &) const
//
// returning a vector of vector_double. Each vector_double contains
// the hessian of the relative fitness component. Each hessian
// being symmetric PaGMO only allow the definition of the diagonal and
// lower triangular compoents in the order defined by the
// hessians sparsity pattern which, by default is dense and is:
// [[(0,0),(1,0), (1,1), (2,0), (2,1), ...], [...], ...]
//
// The default dense hessians sparsity can be overridden by
// adding one optional method
//
// sparsity_pattern hessians_sparsity() const
//
// In this example we explicitly define the sparsity patterns as
// to clarify the above notation.

#include <iostream>
#include <string>
#include <vector>

#include <pagmo/io.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;
struct problem_basic_gh {
    // Mandatory, computes ... well ... the fitness
    vector_double fitness(const vector_double &x) const
    {
        return {x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]};
    }

    // Mandatory, returns the box-bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{-10, -10, -10, -10}, {10, 10, 10, 10}};
    }

    // Optional, computes the gradients. In this simple case only one
    // df0/dx0, df0/dx1, df0/dx2, df0/dx3
    vector_double gradient(const vector_double &x) const
    {
        return {2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3]};
    }

    // Optional. Returns the sparsity of the problem as a sparsity_pattern
    // that is pairs (i,j) indicating that the j-th variable
    // "influences" the i-th component in the fitness.
    // When not implemented a dense problem is assumed.
    sparsity_pattern gradient_sparsity() const
    {
        return {{0, 0}, {0, 1}, {0, 2}, {0, 3}};
    }

    // Optional. Returns the Hessians of the various fitness
    // components fk. That is d^2fk/dxi/dxj. In this case we have only
    // one fitness component, thus we only need one Hessian which is
    // also sparse as most of its components are 0.
    std::vector<vector_double> hessians(const vector_double &) const
    {
        return {{2., 2., 2., 2.}};
    }

    // Optional. Returns the sparsity of the hessians.
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return {{{0, 0}, {1, 1}, {2, 2}, {3, 3}}};
    }

    // Optional, provides a name for the problem overrding the default name
    std::string get_name() const
    {
        return "My Problem with derivatives!!";
    }

    // Optional, provides extra information that will be appended after
    // the default stream operator
    std::string get_extra_info() const
    {
        std::ostringstream s;
        s << "This is a simple toy problem with one fitness, " << '\n';
        s << "no constraint and a fixed dimension of 4."
          << "\n";
        s << "The fitness function gradient and hessians are also implemented"
          << "\n";
        s << "The sparsity of the gradient and hessians is user provided"
          << "\n";
        return s.str();
    }

    // Optional methods-data can also be accessed later via
    // the problem::extract() method
    vector_double best_known() const
    {
        return {0, 0, 0, 0};
    }
};

int main()
{
    // Constructing a problem
    problem p0{problem_basic_gh{}};
    // Streaming to screen the problem
    std::cout << p0 << '\n';
    // Getting its dimensions
    std::cout << "Calling the dimension getter: " << p0.get_nx() << '\n';
    std::cout << "Calling the fitness dimension getter: " << p0.get_nobj() << '\n';

    // Getting the bounds via the pagmo::print eating also std containers
    pagmo::print("Calling the bounds getter: ", p0.get_bounds(), "\n\n");

    // As soon as a problem its created its function evaluation counter
    // is set to zero. Checking its value is easy
    pagmo::print("fevals: ", p0.get_fevals(), "\n");
    // Computing one fitness
    pagmo::print("calling fitness in x=[2,2,2,2]: ", p0.fitness({2, 2, 2, 2}), "\n");
    // The evaluation counter is now ... well ... 1
    pagmo::print("fevals: ", p0.get_fevals(), "\n\n");

    // As soon as a problem its created its gradient evaluation counter
    // is set to zero. Checking its value is easy
    pagmo::print("gevals: ", p0.get_gevals(), "\n");
    // Computing one gradient
    pagmo::print("gradient implementation detected?: ", p0.has_gradient(), '\n');
    pagmo::print("calling gradient in x=[2,2,2,2]: ", p0.gradient({2, 2, 2, 2}), "\n");
    // The evaluation counter is now ... well ... 1
    pagmo::print("gevals: ", p0.get_gevals(), "\n\n");

    // As soon as a problem its created its hessian evaluation counter
    // is set to zero. Checking its value is easy
    pagmo::print("hevals: ", p0.get_hevals(), "\n");
    // Computing one gradient
    pagmo::print("hessians implementation detected?: ", p0.has_hessians(), '\n');
    pagmo::print("calling hessians in x=[2,2,2,2]: ", p0.hessians({2, 2, 2, 2}), "\n");
    // The evaluation counter is now ... well ... 1
    pagmo::print("hevals: ", p0.get_hevals(), "\n\n");

    pagmo::print("Gradient sparsity pattern: ", p0.gradient_sparsity(), "\n");
    pagmo::print("Hessians sparsity pattern: ", p0.hessians_sparsity(), "\n\n");

    // While our problem_basic_gh struct is now hidden inside the pagmo::problem
    // we can still access its methods / data via the extract interface
    pagmo::print("Accessing best_known: ", p0.extract<problem_basic_gh>()->best_known(), "\n");
}
