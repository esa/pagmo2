/*
 *  Implements the NSGA-III multi-objective evolutionary algorithm
 *  as described in http://dx.doi.org/10.1109/TEVC.2013.2281535
 *
 *  Paul Slavin <paul.slavin@manchester.ac.uk>
 */
#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithms/nsga3.hpp>


namespace pagmo{

nsga3::nsga3(unsigned gen, double cr, double eta_c, double m, double eta_m, unsigned seed)
        : gen(gen), cr(cr), eta_c(eta_c), m(m), eta_m(eta_m), seed(seed), reng(seed){
    ;
    // Validate ctor args
}

population nsga3::evolve(population pop) const{
    const auto &prob = pop.get_problem();
    const auto bounds = prob.get_bounds();
    auto dim_i = prob.get_nix();
    auto NP = pop.size();
    auto fevals0 = prob.get_fevals();

    // WIP: Avoid build failure for unused-vars with Werror
    (void)dim_i;
    (void)NP;
    (void)fevals0;

    return pop;
}

}
