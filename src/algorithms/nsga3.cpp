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


std::vector<std::vector<double>> nsga3::translate_objectives(population pop){
    size_t NP = pop.size();
    auto objs = pop.get_f();
    std::vector<double> p_ideal = ideal(objs);
    std::vector<std::vector<double>> translated_objs(NP, {0.0, 0.0, 0.0});

    for(size_t obj=0; obj<3; obj++){
        for(size_t i=0; i<NP; i++){
            translated_objs[i][obj] = objs[i][obj] - p_ideal[obj];
        }
    }

    return translated_objs;
}

// fronts arg is NDS return type
std::vector<size_t> nsga3::find_extreme_points(population pop,
                                               std::vector<std::vector<pop_size_t>> &fronts,
                                               std::vector<std::vector<double>> &translated_objs){
    std::vector<size_t> points;
    size_t nobj = pop.get_problem().get_nobj();

    for(size_t i=0; i<nobj; i++){
        std::vector<double> weights(nobj, 1e-6);
        weights[i] = 1.0;
        double min_asf = std::numeric_limits<double>::max();
        double min_individual = fronts[0].size();
        // Only first front need be considered for extremes
        for(size_t ind=0; ind<fronts[0].size(); ind++){
            // Calculate ASF value for translated objectives
            double asf = achievement(translated_objs[fronts[0][ind]], weights);
            if(asf < min_asf){
                min_asf = asf;
                min_individual = fronts[0][ind];
            }
        }
        points.push_back(min_individual);
    }

    return points;
}

std::vector<double> nsga3::find_intercepts(population pop, std::vector<size_t> &ext_points,
                                           std::vector<std::vector<double>> &translated_objs){
    /*  1. Check duplicate extreme points
     *  2. A = translated objectives of extreme points;  b = [1,1,...] to n_objs
     *  3. Solve Ax = b via Gaussian Elimination
     *  4. Return reciprocals as intercepts
     */

    size_t n_obj = pop.get_problem().get_nobj();
    std::vector<double> b(n_obj, 1.0);
    std::vector<std::vector<double>> A;

    for(size_t i=0; i<ext_points.size(); i++){
        A.push_back(translated_objs[ext_points[i]]);
    }

    // Ax = b
    std::vector<double> x = gaussian_elimination(A, b);

    // Express as intercepts, 1/x
    std::vector<double> intercepts(n_obj, 1.0);
    for(size_t i=0; i<intercepts.size(); i++){
        intercepts[i] = 1.0/x[i];
    }

    return intercepts;
}

//  Equation 4: Note
std::vector<std::vector<double>> nsga3::normalize_objectives(std::vector<std::vector<double>> &translated_objs,
                                                      std::vector<double> &intercepts){
    /*  Algorithm 2, step 7 and Equation 4
     *  Note that Objectives and therefore intercepts
     *  are already translated by ideal point.
     */

    size_t objs = translated_objs[1].size();
    std::vector<std::vector<double>> norm_objs(translated_objs.size(), std::vector<double>(objs));

    for(size_t i=0; i<translated_objs.size(); i++){
        for(size_t obj=0; obj<objs; obj++){
            double intercept_or_eps = std::max(intercepts[obj], std::numeric_limits<double>::epsilon());
            norm_objs[i][obj] = translated_objs[i][obj]/intercept_or_eps;
        }
    }

    return norm_objs;
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

    //std::for_each(shuffle1.begin(), shuffle1.end(), [](const auto& elem){std::cout << elem << " "; });
    //std::cout << std::endl;

    for(decltype(ngen)gen = 1u; gen <= ngen; gen++){
        std::cout << "Generation: " << gen << "/" << ngen << std::endl;
        // Copy existing population
        population popnew(pop);

        // Permute population indices
        std::shuffle(shuffle1.begin(), shuffle1.end(), reng);
        std::shuffle(shuffle2.begin(), shuffle2.end(), reng);
        std::for_each(shuffle1.begin(), shuffle1.end(), [](const auto& elem){std::cout << elem << " "; });
        std::cout << std::endl;

        auto p0 = pop.get_f();
        std::cout << "p0.size(): " << p0.size() << std::endl;
        std::cout << "pop: ";
        for(size_t i=0; i < p0.size(); i++){
            std::cout << i << "\t";
            std::for_each(p0[i].begin(), p0[i].end(), [](const auto& elem){std::cout << elem << ", "; });
            std::cout << std::endl;
        }
        std::vector<double> p_ideal = ideal(p0);
        std::vector<double> p_nadir = nadir(p0);
        std::cout << "ideal: ";
        std::for_each(p_ideal.begin(), p_ideal.end(), [](const auto& elem){std::cout << elem << " ";});
        std::cout << std::endl;
        std::cout << "nadir: ";
        std::for_each(p_nadir.begin(), p_nadir.end(), [](const auto& elem){std::cout << elem << " ";});
        std::cout << std::endl;
        auto fnds_res = fast_non_dominated_sorting(pop.get_f());
        std::cout << "fnds_res tuple_size: " << std::tuple_size<fnds_return_type>::value << std::endl;

        auto nds0 = std::get<0>(fnds_res);
        auto nds1 = std::get<1>(fnds_res);
        auto nds2 = std::get<2>(fnds_res);
        auto nds3 = std::get<3>(fnds_res);
        std::cout << nds0.size() << " " << nds1.size() << " " << nds2.size() << " " << nds3.size() << std::endl;
        std::for_each(nds0[0].begin(), nds0[0].end(), [](const auto& elem){std::cout << elem << " "; });
        std::cout << std::endl;
        std::for_each(nds0[1].begin(), nds0[1].end(), [](const auto& elem){std::cout << elem << " "; });
        std::cout << std::endl;
        std::for_each(nds0[2].begin(), nds0[2].end(), [](const auto& elem){std::cout << elem << " "; });
        std::cout << std::endl;
        std::for_each(nds0[3].begin(), nds0[3].end(), [](const auto& elem){std::cout << elem << " "; });
        std::cout << std::endl;
        std::for_each(nds0[4].begin(), nds0[4].end(), [](const auto& elem){std::cout << elem << " "; });
        std::cout << std::endl;

        //std::cout << std::to_string(nds0[0][0]) << std::endl;
        //std::cout << nds1;
        //std::cout << nds2;
        //std::cout << nds3;
    }

    return pop;
}

}
