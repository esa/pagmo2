#include <cmath>
#include <initializer_list>
#include <iostream>
#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>
#include <pagmo/algorithms/gaco.hpp>


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
    // 1 - Construct a pagmo::problem from our example problem.
    problem prob{problem_v1{}};

    // 2 - Define a algorithm to solve the problem. In this case the Extended Ant Colony Optimization with 1000 generations
    algorithm algo{gaco(1000)};

    // 3 - Initiate the population of 1000 individuals
    population pop{prob, 1000};

    // 4 - Evolve the initialized population
    pop = algo.evolve(pop);

    // 5 - Print the final individuals after
    std::cout << pop;
}

