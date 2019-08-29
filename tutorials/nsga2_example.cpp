#include <iostream>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/dtlz.hpp>

using namespace pagmo;

int main()
{
    // 1 - Instantiate a pagmo problem constructing it from a UDP
    // (user defined problem).
    problem prob{dtlz(1, 10, 2)};

    // 2 - Instantiate a pagmo algorithm
    algorithm algo{nsga2(100)};

    // 3 - Instantiate a population
    population pop{prob, 24};

    // 4 - Evolve the population
    pop = algo.evolve(pop);

    // 5 - Output the population
    std::cout << "The population: \n" << pop;
}
