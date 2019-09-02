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
        return {std::sqrt(dv[1])};
    }
    // Implementation of the box bounds.
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{-0.5, 0}, {1, 8}};
    }
};

int main()
{
    // Construct a pagmo::problem from our example problem.
    problem p{problem_v0{}};

    // Compute the value of the objective function
    // in the point (1, 2).
    std::cout << "Value of the objfun in (1, 2): " << p.fitness({1, 2})[0] << '\n';

    // Print p to screen.
    std::cout << p << '\n';
}
