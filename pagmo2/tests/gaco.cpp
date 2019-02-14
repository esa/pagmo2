#include <iostream>
#include <pagmo/algorithm.hpp>
#include <pagmo/problem.hpp>

#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/problems/rosenbrock.hpp>

using namespace pagmo;
int main()
{
    // Set seed for reproducible results
    pagmo::random_device::set_seed(12345);

    // Algorithm (setting generations to 2000)
    pagmo::algorithm algo{g_aco{2000}};

    // Set the algo to log something at each iteration
    algo.set_verbosity(1);

    // Problem
    pagmo::problem prob{rosenbrock{10}};

    // Population
    pagmo::population pop{prob, 200};

    // Evolve for 2000 generations
    pop = algo.evolve(pop);

    // Print to console
    std::cout << pop << std::endl;

    return 0;
}
