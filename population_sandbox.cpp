#include <string>

#include "include/io.hpp"
#include "include/population.hpp"
#include "include/problem.hpp"
#include "include/problems/hock_schittkowsky_71.hpp"
#include "include/problems/rosenbrock.hpp"
#include "include/utils/constrained.hpp"

using namespace pagmo;


int main()
{
    // Constructing a population
    problem prob{rosenbrock{5}};
    population pop{prob, 4};
    print(pop);
    print(sort_population_con(pop.get_f(), 0u,'\n'));
    print(pop.champion(),'\n');
}
