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


std::vector<ReferencePoint> nsga3::generate_uniform_reference_points(size_t nobjs, size_t divisions) const{
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

population nsga3::evolve(population &pop) const{
    const auto &prob = pop.get_problem();
    const auto bounds = prob.get_bounds();
    auto dim_i = prob.get_nix();
    auto NP = pop.size();

    // Initialize the population

    /* Verify problem characteristics:
     *  - Has multiple objectives
     *  - Is not stochastic
     *  - Has unequal bounds
     *  - No non-linear constraints
     *  - "Appropriate" population size and factors; NP >= num reference directions
     */

    std::vector<vector_double::size_type> best_idx(NP), shuffle1(NP), shuffle2(NP);
    std::pair<vector_double, vector_double> children;

    // Initialise population indices
    std::iota(shuffle1.begin(), shuffle1.end(), vector_double::size_type(0));
    std::iota(shuffle2.begin(), shuffle2.end(), vector_double::size_type(0));

    for(decltype(ngen)gen = 1u; gen <= ngen; gen++){
        // Copy existing population
        population popnew(pop);

        // Permute population indices
        std::shuffle(shuffle1.begin(), shuffle1.end(), reng);
        std::shuffle(shuffle2.begin(), shuffle2.end(), reng);

        /*  1. Generate offspring population Q_t
         *  2. R = P_t U Q_t
         *  3. P_t+1 = selection(R)
         */

        if(gen % 500 == 0){
            std::cout << "Generation: " << gen << "/" << ngen << "\n";
            std::cout << "fevals: " << prob.get_fevals() << "\n";
            std::vector<double> p_ideal = ideal(pop.get_f());
            std::vector<double> p_nadir = nadir(pop.get_f());
            std::cout << "ideal: ";
            std::for_each(p_ideal.begin(), p_ideal.end(), [](const auto& elem){std::cout << elem << " ";});
            std::cout << "\nnadir: ";
            std::for_each(p_nadir.begin(), p_nadir.end(), [](const auto& elem){std::cout << elem << " ";});
            std::cout << std::endl;
        }
        if(gen == ngen){
            auto p0 = pop.get_f();
            std::cout << "pop: ";
            for(size_t i=0; i < p0.size(); i++){
                std::cout << i << "\t[";
                std::for_each(p0[i].begin(), p0[i].end(), [](const auto& elem){std::cout << elem << ", "; });
                std::cout << "]" << std::endl;
            }
        }

        // Offspring generation
        for (decltype(NP) i = 0u; i < NP; i += 4) {
            // We create two offsprings using the shuffled list 1
            decltype(shuffle1) parents1;
            std::sample(shuffle1.begin(), shuffle1.end(), std::back_inserter(parents1),
                        2, std::mt19937{std::random_device{}()});
            children = detail::sbx_crossover_impl(pop.get_x()[parents1[0]], pop.get_x()[parents1[1]], bounds, dim_i,
                                                  cr, eta_c, reng);
            detail::polynomial_mutation_impl(children.first, bounds, dim_i, m, eta_m, reng);
            detail::polynomial_mutation_impl(children.second, bounds, dim_i, m, eta_m, reng);
            // Evaluation via prob ensures feval counter is correctly updated
            auto f1 = prob.fitness(children.first);
            auto f2 = prob.fitness(children.second);
            popnew.push_back(children.first, f1);
            popnew.push_back(children.second, f2);

            // Repeat with the shuffled list 2
            decltype(shuffle2) parents2;
            std::sample(shuffle2.begin(), shuffle2.end(), std::back_inserter(parents2),
                        2, std::mt19937{std::random_device{}()});
            children = detail::sbx_crossover_impl(pop.get_x()[parents2[0]], pop.get_x()[parents2[1]], bounds, dim_i,
                                                  cr, eta_c, reng);
            detail::polynomial_mutation_impl(children.first, bounds, dim_i, m, eta_m, reng);
            detail::polynomial_mutation_impl(children.second, bounds, dim_i, m, eta_m, reng);
            f1 = prob.fitness(children.first);
            f2 = prob.fitness(children.second);
            popnew.push_back(children.first, f1);
            popnew.push_back(children.second, f2);
        } // popnew now contains |P_t|+|R| = 2NP individuals

        // Select NP individuals for next generation
        std::vector<size_t> pop_next = selection(popnew, NP);
        for(population::size_type i = 0; i<NP; ++i){
            pop.set_xf(i, popnew.get_x()[pop_next[i]], popnew.get_f()[pop_next[i]]);
        }
    }
    return pop;
}
/*  Selects members of a population for survival into the next generation
 *  arguments:
 *    population R: The combined parent and offspring populations
 *                  of size 2*N_pop
 *    size_t N_pop: The target population size to return
 *
 */
std::vector<size_t> nsga3::selection(population &R, size_t N_pop) const{

    std::vector<size_t> next;
    next.reserve(N_pop);
    size_t last_front = 0;
    size_t next_size = 0;
    size_t nobj = R.get_problem().get_nobj();

    fnds_return_type nds = fast_non_dominated_sorting(R.get_f());
    auto fronts = std::get<0>(nds);

    while(next_size < N_pop){
        next_size += fronts[last_front++].size();
    }
    // Remove dominated fronts surplus to required N_pop
    fronts.erase(fronts.begin() + last_front, fronts.end());

    // Accept all members of first l-1 fronts
    for(size_t f=0; f<fronts.size()-1; f++){
        for(size_t i=0; i<fronts[f].size(); i++){
            next.push_back(fronts[f][i]);
        }
    }

    if(next.size() == N_pop){
        return next;
    }

    auto translated_objectives = translate_objectives(R);
    auto ext_points = find_extreme_points(R, fronts, translated_objectives);
    auto intercepts = find_intercepts(R, ext_points, translated_objectives);
    auto norm_objs = normalize_objectives(translated_objectives, intercepts);
    std::vector<ReferencePoint> rps = generate_uniform_reference_points(nobj, 12 /* parameter */);
    associate_with_reference_points(rps, norm_objs, fronts);

    // For visualisation
    /*
    for(auto &rp: rps){
        using std::cout, std::endl;
        // std::for_each(rp.begin(), rp.end(), [](const auto& elem){std::cout << elem << " ";});
        cout << "[" << rp[0] << ", " << rp[1] << ", " << rp[2] << "],\n";
    }
    std::cout << std::endl;
    */

    // Apply RP selection to final front until N_pop reached
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

// Object serialization
template <typename Archive>
void nsga3::serialize(Archive &ar, unsigned int) {
    detail::archive(ar, ngen, cr, eta_c, m, eta_m, seed);
}

}  // namespace pagmo
