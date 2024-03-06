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
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/genetic_operators.hpp>
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


std::vector<std::vector<double>> nsga3::translate_objectives(population pop) const{
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
                                               std::vector<std::vector<double>> &translated_objs) const{
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
                                           std::vector<std::vector<double>> &translated_objs) const{
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
                                                      std::vector<double> &intercepts) const{
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

population nsga3::evolve(population &pop){
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
    vector_double::size_type parent1_idx, parent2_idx;
    std::pair<vector_double, vector_double> children;

    (void)best_idx;

    // Initialise population indices
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

        /*  1. Generate offspring population Q_t
         *  2. R = P_t U Q_t
         *  3. P_t+1 = selection(R)
         */

        std::cout << "fevals: " << prob.get_fevals() << std::endl;
        std::vector<double> p_ideal = ideal(pop.get_f());
        std::vector<double> p_nadir = nadir(pop.get_f());
        std::cout << "ideal: ";
        std::for_each(p_ideal.begin(), p_ideal.end(), [](const auto& elem){std::cout << elem << " ";});
        std::cout << std::endl;
        std::cout << "nadir: ";
        std::for_each(p_nadir.begin(), p_nadir.end(), [](const auto& elem){std::cout << elem << " ";});
        std::cout << std::endl;

        auto fnds_res = fast_non_dominated_sorting(pop.get_f());
        auto nd_fronts = std::get<0>(fnds_res);  // Non-dominated fronts
        auto nd_rank = std::get<3>(fnds_res);
        vector_double pop_cd(NP);         // crowding distances of the whole population
                                          //
        for (const auto &front_idxs : nd_fronts) {
            if (front_idxs.size() == 1u) { // handles the case where the front has collapsed to one point
                pop_cd[front_idxs[0]] = std::numeric_limits<double>::infinity();
            } else {
                if (front_idxs.size() == 2u) { // handles the case where the front has collapsed to one point
                    pop_cd[front_idxs[0]] = std::numeric_limits<double>::infinity();
                    pop_cd[front_idxs[1]] = std::numeric_limits<double>::infinity();
                } else {
                    std::vector<vector_double> front;
                    for (auto idx : front_idxs) {
                        front.push_back(pop.get_f()[idx]);
                    }
                    auto cd = crowding_distance(front);
                    for (decltype(cd.size()) i = 0u; i < cd.size(); ++i) {
                        pop_cd[front_idxs[i]] = cd[i];
                    }
                }
            }
        }

        for (decltype(NP) i = 0u; i < NP; i += 4) {
            // We create two offsprings using the shuffled list 1
            parent1_idx = detail::mo_tournament_selection_impl(shuffle1[i], shuffle1[i + 1], nd_rank, pop_cd, reng);
            parent2_idx = detail::mo_tournament_selection_impl(shuffle1[i + 2], shuffle1[i + 3], nd_rank, pop_cd, reng);
            children = detail::sbx_crossover_impl(pop.get_x()[parent1_idx], pop.get_x()[parent2_idx], bounds, dim_i,
                                                  cr, eta_c, reng);
            detail::polynomial_mutation_impl(children.first, bounds, dim_i, m, eta_m, reng);
            detail::polynomial_mutation_impl(children.second, bounds, dim_i, m, eta_m, reng);
            // we use prob to evaluate the fitness so
            // that its feval counter is correctly updated
            auto f1 = prob.fitness(children.first);
            auto f2 = prob.fitness(children.second);
            popnew.push_back(children.first, f1);
            popnew.push_back(children.second, f2);

            // We repeat with the shuffled list 2
            parent1_idx = detail::mo_tournament_selection_impl(shuffle2[i], shuffle2[i + 1], nd_rank, pop_cd, reng);
            parent2_idx = detail::mo_tournament_selection_impl(shuffle2[i + 2], shuffle2[i + 3], nd_rank, pop_cd, reng);
            children = detail::sbx_crossover_impl(pop.get_x()[parent1_idx], pop.get_x()[parent2_idx], bounds, dim_i,
                                                  cr, eta_c, reng);
            detail::polynomial_mutation_impl(children.first, bounds, dim_i, m, eta_m, reng);
            detail::polynomial_mutation_impl(children.second, bounds, dim_i, m, eta_m, reng);
            // we use prob to evaluate the fitness so
            // that its feval counter is correctly updated
            f1 = prob.fitness(children.first);
            f2 = prob.fitness(children.second);
            popnew.push_back(children.first, f1);
            popnew.push_back(children.second, f2);
        } // popnew now contains 2NP individuals

        std::vector<size_t> pop_next = selection(popnew, NP);
        //for(size_t i=0; i<pop_next.size(); i++){
        //    std::cout << "pop_next[" << i << "] = " << pop_next[i] << std::endl;
        //}
        for(population::size_type i = 0; i<NP; ++i){
            pop.set_xf(i, popnew.get_x()[pop_next[i]], popnew.get_f()[pop_next[i]]);
        }

        /*
        auto fnds_res = fast_non_dominated_sorting(pop.get_f());
        auto ndf = std::get<0>(fnds_res);  // Non-dominated fronts
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
        */
        //std::cout << std::to_string(nds0[0][0]) << std::endl;
        //std::cout << nds1;
        //std::cout << nds2;
        //std::cout << nds3;
    }

    return pop;
}
/*  Selects members of a population for survival into the next generation
 *  arguments:
 *    population Q: The combined parent and offspring populations
 *                  of size 2*N_pop
 *    size_t N_pop: The target population size to return
 *
 */
std::vector<size_t> nsga3::selection(population &Q, size_t N_pop){

    std::vector<size_t> next;
    size_t last_front = 0;
    size_t next_size = 0;
    size_t nobj = Q.get_problem().get_nobj();

    fnds_return_type nds = fast_non_dominated_sorting(Q.get_f());
    auto fronts = std::get<0>(nds);
    /*
    std::for_each(fronts[0].begin(), fronts[0].end(), [](const auto& elem){std::cout << elem << " "; });
    std::cout << std::endl;
    std::for_each(fronts[1].begin(), fronts[1].end(), [](const auto& elem){std::cout << elem << " "; });
    std::cout << std::endl;
    std::for_each(fronts[2].begin(), fronts[2].end(), [](const auto& elem){std::cout << elem << " "; });
    std::cout << std::endl;
    std::for_each(fronts[3].begin(), fronts[3].end(), [](const auto& elem){std::cout << elem << " "; });
    std::cout << std::endl;
    std::for_each(fronts[4].begin(), fronts[4].end(), [](const auto& elem){std::cout << elem << " "; });
    std::cout << std::endl;
    */

    while(next_size < N_pop){
        next_size += fronts[last_front++].size();
    }
    fronts.erase(fronts.begin() + last_front, fronts.end());

    /*  This won't work: need to build a
     *  temp map of points with size equal
     *  to N_pop, return this and then assign
     *  in caller with below.
     *
    population::size_type idx = 0;
    for(const auto &front: fronts){
        for(size_t i=0; i<front.size(); i++){
            next.set_x(idx++, Q_x[front[i]]);
            //next.push_back(Q_x[front[i]]);
        }
    }*/

    for(size_t f=0; f<fronts.size()-1; f++){
        for(size_t i=0; i<fronts[f].size(); i++){
            next.push_back(fronts[f][i]);
        }
    }

    if(next.size() == N_pop){
        return next;
    }

    auto translated_objectives = translate_objectives(Q);
    auto ext_points = find_extreme_points(Q, fronts, translated_objectives);
    auto intercepts = find_intercepts(Q, ext_points, translated_objectives);
    auto norm_objs = normalize_objectives(translated_objectives, intercepts);
    std::vector<ReferencePoint> rps = generate_uniform_reference_points(nobj, 12 /* parameter */);
    associate_with_reference_points(rps, norm_objs, fronts);

    //std::vector<vector_double> Q_x = Q.get_x();
    while(next.size() < N_pop){
        size_t min_rp_idx = identify_niche_point(rps);
        int selected_idx = rps[min_rp_idx].select_member();
        if(selected_idx < 0){
            rps.erase(rps.begin() + min_rp_idx);
        }else{
            rps[min_rp_idx].increment_members();
            rps[min_rp_idx].remove_candidate(selected_idx);
            next.push_back(selected_idx);
        }
    }

    return next;
}

}
