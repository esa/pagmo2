#include <cmath>
#include <initializer_list>
#include <iostream>
#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

// Our simple example problem, version 1.
struct problem_v1 {
    // Number of equality constraints.
    vector_double::size_type get_nec() const
    {
        return 1;
    }
    // Number of inequality constraints.
    vector_double::size_type get_nic() const
    {
        return 1;
    }
    // Implementation of the objective function.
    vector_double fitness(const vector_double &dv) const
    {
        return {
            dv[0] * dv[3] * (dv[0] + dv[1] + dv[2]) + dv[2],                     // objfun
            dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2] + dv[3] * dv[3] - 40., // equality con.
            25. - dv[0] * dv[1] * dv[2] * dv[3]                                  // inequality con.
        };
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
    problem p{problem_v1{}};

    // Compute the value of the objective function, equality and
    // inequality constraints in the point (1, 2, 3, 4).
    const auto fv = p.fitness({1, 2, 3, 4});
    std::cout << "Value of the objfun in (1, 2, 3, 4): " << fv[0] << '\n';
    std::cout << "Value of the eq. constraint in (1, 2, 3, 4): " << fv[1] << '\n';
    std::cout << "Value of the ineq. constraint in (1, 2, 3, 4): " << fv[2] << '\n';

    // Fetch the lower/upper bounds for the first variable.
    std::cout << "Lower bounds: [" << p.get_lb()[0] << "]\n";
    std::cout << "Upper bounds: [" << p.get_ub()[0] << "]\n\n";

    // Print p to screen.
    std::cout << p << '\n';
}
