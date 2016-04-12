// In this tutorial we learn how to implement gradients and hessian
// in case whe have them. The problem is to minimize:
// f = x1^2 + x2^2 + x3^2 + x4^2 in the bounds:
// -10 <= xi <= 10
//

// All we need to do is to implement a struct (or class) having the
// following mandatory methods: 
//
// vector_double fitness(const vector_double &)
// vector_double::size_type get_n() const
// vector_double::size_type get_nf() const
// std::pair<vector_double, vector_double> get_bounds() const
//
// And add a method:
// vector_double gradient(const vector_double &x)

#include <string>

#include "include/io.hpp"
#include "include/problem.hpp"
#include "include/types.hpp"



using namespace pagmo;
struct example0_g
{
    // Mandatory, computes ... well ... the fitness
    vector_double fitness(const vector_double &x) const
    {
        return {x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]};
    }

    // Optional, computes the gradient. In this simple case
    // df0/dx0, df0/dx1, df0/dx2, df0/dx3
    vector_double gradient(const vector_double &x) const
    {
        return {2 * x[0],2 * x[1], 2 * x[2], 2 * x[3]};
    }

    // Optional. Returns the sparsity of the problem as a sparsity_pattern
    // that is a std::vector<std::pair<long,long>> containing pairs
    // (i,j) indicating that the j-th variable "influences" the i-th component
    // in the fitness. When not implemented a dense problem is assumed
    sparsity_pattern gradient_sparsity() const
    {
        return {{0,0},{0,1},{0,2},{0,3}};
    }

    // Optional, computes the Hessians of the various fitness
    // components. That is d^2fk/dxidxj. In this case we have only
    // one fitness component, thus we only need one Hessian which is
    // also sparse as most of its components are 0.
    std::vector<vector_double> hessians(const vector_double &) const
    {
        return {{2.,2.,2.,2.}};
    }

    // Optional, computes the sparsity of the hessians.
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return {{{0,0},{1,1},{2,2},{3,3}}};
    }

    // Mandatory, returns the dimension of the decision vector,
    // in this case fixed to 4
    vector_double::size_type get_n() const
    {
        return 4u;
    }

    // Mandatory, returns the dimension of the decision vector,
    // in this case fixed to 1 (single objective)
    vector_double::size_type get_nf() const
    {
        return 1u;
    }
    
    // Mandatory, returns the box-bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{-10,-10,-10,-10},{10,10,10,10}};
    }

    // Optional, provides a name for the problem overrding the default name
    std::string get_name() const
    {   
        return std::string("My Problem");
    }

    // Optional, provides extra information that will be appended after
    // the default stream operator
    std::string extra_info() const {
        std::ostringstream s;
        s << "This is a simple toy problem with one fitness, " << '\n';
        s << "no constraint and a fixed dimension of 4." << "\n";
        s << "The fitness function gradients are also implemented" << "\n";
        return s.str();
    }
    
    // Optional methods-data can also be accessed later via 
    // the problem::extract() method
    std::vector<vector_double> best_known() const
    {
        return {{0,0,0,0}};
    }
};

int main()
{
    // Constructing a problem
    problem p0{example0_g{}};
    // Streaming to screen the problem
    std::cout << p0 << '\n';
    // Getting its dimensions
    std::cout << "Calling the dimension getter: " << p0.get_n() << '\n';
    std::cout << "Calling the fitness dimension getter: " << p0.get_nf() << '\n';

    // Getting the bounds via the pagmo::print eating also std containers
    pagmo::print("Calling the bounds getter: ", p0.get_bounds(), "\n\n");

    // As soon as a problem its created its function evaluation counter
    // is set to zero. Checking its value is easy
    pagmo::print("fevals: ", p0.get_fevals(), "\n");
    // Computing one fitness
    pagmo::print("calling fitness in x=[2,2,2,2]: ", p0.fitness(vector_double{2,2,2,2}), "\n");
    // The evaluation counter is now ... well ... 1
    pagmo::print("fevals: ", p0.get_fevals(), "\n\n");

    // As soon as a problem its created its gradient evaluation counter
    // is set to zero. Checking its value is easy
    pagmo::print("gevals: ", p0.get_gevals(), "\n");
    // Computing one gradient
    pagmo::print("gradient implementation detected?: ", p0.has_gradient(), '\n');
    pagmo::print("calling gradient in x=[2,2,2,2]: ", p0.gradient(vector_double{2,2,2,2}), "\n");
    // The evaluation counter is now ... well ... 1
    pagmo::print("gevals: ", p0.get_gevals(), "\n\n");

    pagmo::print("Sparsity pattern: ", p0.gradient_sparsity(), "\n\n");

    // While our example0 struct is now hidden inside the pagmo::problem
    // we can still access its methods / data via the extract interface
    pagmo::print("Accessing best_known: ", p0.extract<example0_g>()->best_known(), "\n");
}
