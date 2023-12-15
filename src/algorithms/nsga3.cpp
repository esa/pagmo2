/*
 *  Implements the NSGA-III multi-objective evolutionary algorithm
 *  as described in http://dx.doi.org/10.1109/TEVC.2013.2281535
 *
 *  Paul Slavin <paul.slavin@manchester.ac.uk>
 */
#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithms/nsga3.hpp>
#include <pagmo/utils/multi_objective.hpp>  // fast_non_dominated_sorting
#include <pagmo/utils/reference_point.hpp>  // ReferencePoint


namespace pagmo{

nsga3::nsga3(unsigned gen, double cr, double eta_c, double m, double eta_m, unsigned seed)
        : ngen(gen), cr(cr), eta_c(eta_c), m(m), eta_m(eta_m), seed(seed), reng(seed){
    ;
    // Validate ctor args
}

std::vector<ReferencePoint> nsga3::generate_uniform_reference_points(size_t nobjs, size_t divisions){
    ReferencePoint rp(nobjs);
    if(!refpoints.empty()){
        refpoints.clear();
    }
    refpoints = generate_reference_point_level(rp, divisions, 0, divisions);
    return refpoints;
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

    // Initialize the population

    /* Verify problem characteristics:
     *  - Has multiple objectives
     *  - Is not stochastic
     *  - Has unequal bounds
     *  - No non-linear constraints
     *  - "Appropriate" population size and factors; NP >= num reference directions
     */

    std::vector<vector_double::size_type> best_idx(NP), shuffle1(NP), shuffle2(NP);
    //vector_double::size_type parent1_idx, parent2_idx;
    //std::pair<vector_double, vector_double> children;

    (void)best_idx;

    std::iota(shuffle1.begin(), shuffle1.end(), vector_double::size_type(0));
    std::iota(shuffle2.begin(), shuffle2.end(), vector_double::size_type(0));

    std::for_each(shuffle1.begin(), shuffle1.end(), [](const auto& elem){std::cout << elem << " "; });
    std::cout << std::endl;

    for(decltype(ngen)gen = 1u; gen <= ngen; gen++){
        std::cout << "Generation: " << gen << "/" << ngen << std::endl;
        // Copy existing population
        population popnew(pop);

        // Permute population indices
        std::shuffle(shuffle1.begin(), shuffle1.end(), reng);
        std::shuffle(shuffle2.begin(), shuffle2.end(), reng);
        std::for_each(shuffle1.begin(), shuffle1.end(), [](const auto& elem){std::cout << elem << " "; });
        std::cout << std::endl;

        auto fnds_res = fast_non_dominated_sorting(pop.get_f());

        auto nds0 = std::get<0>(fnds_res);
        auto nds1 = std::get<1>(fnds_res);
        auto nds2 = std::get<2>(fnds_res);
        auto nds3 = std::get<3>(fnds_res);
        std::for_each(nds0[0].begin(), nds0[0].end(), [](const auto& elem){std::cout << elem << " "; });
        std::cout << std::endl;

        //std::cout << std::to_string(nds0[0][0]) << std::endl;
        //std::cout << nds1;
        //std::cout << nds2;
        //std::cout << nds3;
    }

    return pop;
}

}
