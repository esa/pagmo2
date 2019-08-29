#include <iostream>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/schwefel.hpp>

using namespace pagmo;

int main()
{
    // 1 - Instantiate a pagmo problem constructing it from a UDP
    // (user defined problem).
    problem prob{schwefel(30)};

    // 2 - Instantiate a pagmo algorithm
    algorithm algo{sade(100)};

    // 3 - Instantiate an archipelago with 16 islands having each 20 individuals
    archipelago archi{16, algo, prob, 20};

    // 4 - Run the evolution in parallel on the 16 separate islands 10 times.
    archi.evolve(10);

    // 5 - Wait for the evolutions to be finished
    archi.wait_check();

    // 6 - Print the fitness of the best solution in each island
    for (const auto &isl : archi) {
        std::cout << isl.get_population().champion_f()[0] << '\n';
    }
}
