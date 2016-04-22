#include <string>

#include "include/io.hpp"
#include "include/population.hpp"
#include "include/problem.hpp"
#include "include/problems/hock_schittkowsky_71.hpp"

using namespace pagmo;


int main()
{
    // Constructing a population
    problem prob{hock_schittkowsky_71{}};
    population pop{prob, 10};
    print(pop);
    
}
