#include <cmath>
#include <initializer_list>
#include <iostream>
#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

// Our simple example problem, version 0.
struct problem_v0 {
    // Implementation of the objective function.
    vector_double fitness(const vector_double &dv) const
    {
        return {dv[0] * dv[3] * (dv[0] + dv[1] + dv[2]) + dv[2]};
    }
    // Implementation of the box bounds.
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{1., 1., 1., 1.}, {5., 5., 5., 5.}};
    }
};

int main()
{
    // Construct a pagmo::problem from our example problem.
    problem p{problem_v0{}};

    // Compute the value of the objective function
    // in the point (1, 2, 3, 4).
    std::cout << "Value of the objfun in (1, 2, 3, 4): " << p.fitness({1, 2, 3, 4})[0] << '\n';

    // Fetch the lower/upper bounds for the first variable.
    std::cout << "Lower bounds: [" << p.get_lb()[0] << "]\n";
    std::cout << "Upper bounds: [" << p.get_ub()[0] << "]\n\n";

    // Print p to screen.
    std::cout << p << '\n';
}
