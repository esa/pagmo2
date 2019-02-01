#include <iostream>
#include "pagmo/algorithms/de1220.hpp"
#include "pagmo/algorithms/pso.hpp"
#include "pagmo/algorithms/de.hpp"
#include "pagmo/algorithms/sga.hpp"
#include "pagmo/algorithms/sade.hpp"
#include "pagmo/island.hpp"
#include "pagmo/problem.hpp"
#include "Problems/himmelblau.h"
#include "Problems/applicationOutput.h"
#include "Problems/saveOptimizationResults.h"


#include <pagmo/algorithms/GACO.hpp>
#include <pagmo/problems/rosenbrock.hpp>

using namespace pagmo;
int main( )
{
    using namespace tudat_pagmo_applications;

    //Set seed for reproducible results
    pagmo::random_device::set_seed( 12345 );


    // Algorithm (setting generations to 100)
    pagmo::algorithm algo{ g_aco {30} };

    // Set the algo to log something at each iteration
    algo.set_verbosity(1);

    // Problem
    pagmo::problem prob{rosenbrock{10}};

    // Population
    pagmo::population pop{prob, 1000};

    // Evolve for 100 generations
    pop = algo.evolve(pop);

    // Print to console
    std::cout << pop << std::endl;

   //pagmo::algorithm algo{ pagmo::de( ) };

    return 0;

}
