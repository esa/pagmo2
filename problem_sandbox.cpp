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
// vector_double::size_type get_nobjz() const
// std::pair<vector_double, vector_double> get_bounds() const
//
// And add a method:
// vector_double gradient(const vector_double &x)

#include <string>

#include "include/io.hpp"
#include "include/problem.hpp"
#include "include/problems/null.hpp"
#include "include/problems/translate.hpp"
#include "include/types.hpp"

using namespace pagmo;

int main()
{
    // Constructing a problem
    problem p0{translate{null_problem{}, {1.}}};
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
    pagmo::print("calling fitness in x=[2,2,2,2]: ", p0.fitness({2}), "\n");
    // The evaluation counter is now ... well ... 1
    pagmo::print("fevals: ", p0.get_fevals(), "\n\n");

    // As soon as a problem its created its gradient evaluation counter
    // is set to zero. Checking its value is easy
    pagmo::print("gevals: ", p0.get_gevals(), "\n");
    // Computing one gradient
    pagmo::print("gradient implementation detected?: ", p0.has_gradient(), '\n');
    pagmo::print("calling gradient in x=[2,2,2,2]: ", p0.gradient({2}), "\n");
    // The evaluation counter is now ... well ... 1
    pagmo::print("gevals: ", p0.get_gevals(), "\n\n");

    // As soon as a problem its created its hessian evaluation counter
    // is set to zero. Checking its value is easy
    pagmo::print("hevals: ", p0.get_hevals(), "\n");
    // Computing one gradient
    pagmo::print("hessians implementation detected?: ", p0.has_hessians(), '\n');
    pagmo::print("calling hessians in x=[2,2,2,2]: ", p0.hessians({2}), "\n");
    // The evaluation counter is now ... well ... 1
    pagmo::print("hevals: ", p0.get_hevals(), "\n\n");

    pagmo::print("Gradient sparsity pattern: ", p0.gradient_sparsity(), "\n");
    pagmo::print("Hessians sparsity pattern: ", p0.hessians_sparsity(), "\n\n");

    // While our example0 struct is now hidden inside the pagmo::problem
    // we can still access its methods / data via the extract interface
    auto best_x = p0.extract<null>()->best_known();
    pagmo::print("Accessing best_known: ", best_x, "\n");

    // Evaluating fitness in best_known
    pagmo::print("calling fitness in best_x: ", p0.fitness(best_x), "\n");
}
