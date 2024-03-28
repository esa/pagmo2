/*
 *  Implements the NSGA-III multi-objective evolutionary algorithm
 *  as described in http://dx.doi.org/10.1109/TEVC.2013.2281535
 *
 *  Paul Slavin <paul.slavin@manchester.ac.uk>
 */
#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga3.hpp>
#include <pagmo/io.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/genetic_operators.hpp>
#include <pagmo/utils/multi_objective.hpp>  // fast_non_dominated_sorting
#include <pagmo/utils/reference_point.hpp>  // ReferencePoint
#include <pagmo/s11n.hpp>

#include <boost/serialization/optional.hpp>


namespace pagmo{

nsga3::nsga3(unsigned gen, double cr, double eta_c, double mut, double eta_mut,
             size_t divisions, unsigned seed, bool use_memory)
        : ngen(gen), cr(cr), eta_c(eta_c), mut(mut), eta_mut(eta_mut),
          divisions(divisions), seed(seed), use_memory(use_memory), reng(seed){
    // Validate ctor args
    if(cr < 0.0 || cr > 1.0){
        pagmo_throw(std::invalid_argument, "The crossover probability must be in the range [0, 1], while a value of "
                                           + std::to_string(cr) + " was detected");
    }
    if(mut < 0.0 || mut > 1.0){
        pagmo_throw(std::invalid_argument, "The mutation probability must be in the range [0, 1], while a value of "
                                           + std::to_string(mut) + " was detected");
    }
    if(eta_c < 1.0 || eta_c > 100.0){
        pagmo_throw(std::invalid_argument, "The distribution index for crossover must be in the range [1, 100], "
                                           "while a value of " + std::to_string(eta_c) + " was detected");
    }
    if(eta_mut < 1.0 || eta_mut > 100.0){
        pagmo_throw(std::invalid_argument, "The distribution index for mutation must be in [1, 100], "
                                           "while a value of " + std::to_string(eta_mut) + " was detected");
    }
    // See Deb. Section V, Table I
    if(divisions < 1){
        pagmo_throw(std::invalid_argument, "Invalid <divisions> argument: " + std::to_string(divisions) + ". "
                                           "Number of reference point divisions per objective must be positive");
    }

    /*  Initialise the global pagmo::random_device with our seed.
     *  This ensures the choose_random_element template function
     *  makes deterministic choices using std::sample.
     */
    random_device::set_seed(seed);
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
    size_t nobj = pop.get_problem().get_nobj();
    auto objs = pop.get_f();
    std::vector<double> p_ideal;
    if(has_memory()){
        decltype(objs) combined{memory.v_ideal};
        if(memory.v_ideal.size() != 0){  // i.e. not first gen
            combined.insert(combined.end(), objs.begin(), objs.end());
            p_ideal = ideal(combined);
        }else{
            p_ideal = ideal(objs);
        }
        memory.v_ideal = p_ideal;
    }else{
        p_ideal = ideal(objs);
    }
    std::vector<std::vector<double>> translated_objs(NP, std::vector<double>(nobj));

    for(size_t obj=0; obj<nobj; obj++){
        for(size_t i=0; i<NP; i++){
            translated_objs[i][obj] = objs[i][obj] - p_ideal[obj];
        }
    }

    return translated_objs;
}

// fronts arg is NDS return type
std::vector<std::vector<double>> nsga3::find_extreme_points(population pop,
                                               std::vector<std::vector<pop_size_t>> &fronts,
                                               std::vector<std::vector<double>> &translated_objs) const{
    std::vector<std::vector<double>> points;
    size_t nobj = pop.get_problem().get_nobj();

    for(size_t i=0; i<nobj; i++){
        std::vector<double> weights(nobj, 1e-6);
        weights[i] = 1.0;
        double min_asf = std::numeric_limits<double>::max();
        std::vector<double> min_obj{};

        if(has_memory()){
            if(memory.v_extreme.size() == 0){
                for(size_t idx=0; idx<nobj ; idx++){
                    memory.v_extreme.push_back(std::vector<double>(nobj, {}));
                }
            }else{
                for(size_t p=0; p<memory.v_extreme.size(); p++){
                    double asf = achievement(memory.v_extreme[p], weights);
                    if(asf < min_asf){
                        min_asf = asf;
                        min_obj = memory.v_extreme[p];
                    }
                }
            }
        }

        // Only first front need be considered for extremes
        for(size_t ind=0; ind<fronts[0].size(); ind++){
            // Calculate ASF value for translated objectives
            double asf = achievement(translated_objs[fronts[0][ind]], weights);
            if(asf < min_asf){
                min_asf = asf;
                min_obj = translated_objs[fronts[0][ind]];
            }
        }
        points.push_back(std::vector<double>(min_obj));
        if(has_memory()){
            memory.v_extreme[i] = std::vector<double>(min_obj);
        }
    }


    return points;
}

std::vector<double> nsga3::find_intercepts(population pop, std::vector<std::vector<double>> &ext_points) const{
    /*  1. Check duplicate extreme points
     *  2. A = translated objectives of extreme points;  b = [1,1,...] to n_objs
     *  3. Solve Ax = b via Gaussian Elimination
     *  4. Return reciprocals as intercepts
     *  NB Duplicate ext_points (singular matrix) and
     *  negative intercepts fall back to nadir values.
     */

    size_t n_obj = pop.get_problem().get_nobj();
    std::vector<double> b(n_obj, 1.0);
    std::vector<double> intercepts(n_obj, 1.0);
    std::vector<std::vector<double>> A;
    bool fallback_to_nadir = false;

    for(size_t p=0; !fallback_to_nadir && p<ext_points.size()-1; p++){
        for(size_t q=p+1; !fallback_to_nadir && q<ext_points.size(); q++){
            for(size_t r=0; r<n_obj; r++){
                fallback_to_nadir = (ext_points[p][r] == ext_points[q][r]);
                if(fallback_to_nadir){
                    break;
                }
            }
        }
    }

    if(!fallback_to_nadir){
        for(size_t i=0; i<ext_points.size(); i++){
            A.push_back(ext_points[i]);
        }

        // Ax = b
        std::vector<double> x = gaussian_elimination(A, b);

        // Express as intercepts, 1/x
        for(size_t i=0; i<intercepts.size(); i++){
            intercepts[i] = 1.0/x[i];
            if(x[i] < 0.0){
                fallback_to_nadir = true;
                break;
            }
        }
    }

    if(fallback_to_nadir){
        auto objs = pop.get_f();
        std::vector<double> v_nadir;
        if(has_memory()){
            decltype(objs) combined{memory.v_nadir};
            if(memory.v_nadir.size() != 0){
                combined.insert(combined.end(), objs.begin(), objs.end());
                v_nadir = nadir(combined);
            }else{
                v_nadir = nadir(objs);
            }
            memory.v_nadir = v_nadir;
        }else{
            v_nadir = nadir(objs);
        }
        for(size_t i=0; i<intercepts.size(); i++){
            intercepts[i] = v_nadir[i];
        }
    }

    return intercepts;
}

std::vector<std::vector<double>> nsga3::normalize_objectives(std::vector<std::vector<double>> &translated_objs,
                                                      std::vector<double> &intercepts) const{
    /*  Algorithm 2, step 7 and Equation 4
     *  Note that Objectives and therefore intercepts
     *  are already translated by ideal point.
     */

    size_t nobj = translated_objs[1].size();
    std::vector<std::vector<double>> norm_objs(translated_objs.size(), std::vector<double>(nobj));

    for(size_t i=0; i<translated_objs.size(); i++){
        for(size_t idx=0; idx<nobj; idx++){
            double intercept_or_eps = std::max(intercepts[idx], std::numeric_limits<double>::epsilon());
            norm_objs[i][idx] = translated_objs[i][idx]/intercept_or_eps;
        }
    }

    return norm_objs;
}

population nsga3::evolve(population pop) const{
    const auto &prob = pop.get_problem();
    const auto bounds = prob.get_bounds();
    const auto fevals0 = prob.get_fevals();
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
    if (detail::some_bound_is_equal(prob)) {
        pagmo_throw(std::invalid_argument, "Lower and upper bounds are equal, " + get_name() +
                    " requires these to be different");
    }
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument,
                    get_name() + " algorithm cannot operate on stochastic problems.");
    }
    if (prob.get_nc() != 0u) {
        pagmo_throw(std::invalid_argument, "Non-linear constraints detected in " + prob.get_name() + " instance. "
                    + get_name() + " cannot deal with them.");
    }
    if (prob.get_nf() < 2u) {
        pagmo_throw(std::invalid_argument, "This is a multiobjective algorithm, while number of objectives detected in "
                    + prob.get_name() + " is " + std::to_string(prob.get_nf()));
    }
    if (NP < 5u || (NP % 4 != 0u)) {
        pagmo_throw(std::invalid_argument,
                    "NSGA-III requires a population greater than 5 and which is divisible by 4."
                    "Detected input population size is: " + std::to_string(NP));
    }
    size_t num_rps = n_choose_k(prob.get_nf() + divisions - 1, divisions);
    if(NP <= num_rps){
        pagmo_throw(std::invalid_argument,
                    "Population size must exceed number of reference points. NP = "
                    + std::to_string(NP) + " while " + std::to_string(divisions) + " divisions for "
                    "reference points gives a total of " + std::to_string(num_rps) + " points.");
    }

    m_log.clear();

    std::vector<vector_double::size_type> shuffle1(NP), shuffle2(NP);
    std::pair<vector_double, vector_double> children;
    size_t count{1u};

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

        if(m_verbosity > 0u){
            std::vector<double> p_ideal = ideal(pop.get_f());
            if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                // We compute the ideal point
                // Every 50 lines print the column names
                if (count % 50u == 1u) {
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:");
                    for (decltype(p_ideal.size()) i = 0u; i < p_ideal.size(); ++i) {
                        if (i >= 5u) {
                            print(std::setw(15), "... :");
                            break;
                        }
                        print(std::setw(15), "ideal" + std::to_string(i + 1u) + ":");
                    }
                    print('\n');
                }
                print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0);
                for (decltype(p_ideal.size()) i = 0u; i < p_ideal.size(); ++i) {
                    if (i >= 5u) {
                        break;
                    }
                    print(std::setw(15), p_ideal[i]);
                }
                print('\n');
                ++count;
            }
            m_log.emplace_back(gen, prob.get_fevals() - fevals0, p_ideal);
        }

        // Offspring generation
        for (decltype(NP) i = 0; i < NP; i += 4) {
            // We create two offsprings using the shuffled list 1
            decltype(shuffle1) parents1;
            std::sample(shuffle1.begin(), shuffle1.end(), std::back_inserter(parents1), 2, std::mt19937{reng()});
            children = detail::sbx_crossover_impl(pop.get_x()[parents1[0]], pop.get_x()[parents1[1]], bounds, dim_i,
                                                  cr, eta_c, reng);
            detail::polynomial_mutation_impl(children.first, bounds, dim_i, mut, eta_mut, reng);
            detail::polynomial_mutation_impl(children.second, bounds, dim_i, mut, eta_mut, reng);
            // Evaluation via prob ensures feval counter is correctly updated
            auto f1 = prob.fitness(children.first);
            auto f2 = prob.fitness(children.second);
            popnew.push_back(children.first, f1);
            popnew.push_back(children.second, f2);

            // Repeat with the shuffled list 2
            decltype(shuffle2) parents2;
            std::sample(shuffle2.begin(), shuffle2.end(), std::back_inserter(parents2), 2, std::mt19937{reng()});
            children = detail::sbx_crossover_impl(pop.get_x()[parents2[0]], pop.get_x()[parents2[1]], bounds, dim_i,
                                                  cr, eta_c, reng);
            detail::polynomial_mutation_impl(children.first, bounds, dim_i, mut, eta_mut, reng);
            detail::polynomial_mutation_impl(children.second, bounds, dim_i, mut, eta_mut, reng);
            f1 = prob.fitness(children.first);
            f2 = prob.fitness(children.second);
            popnew.push_back(children.first, f1);
            popnew.push_back(children.second, f2);
        } // popnew now contains |P_t|+|R| = 2NP individuals

        // Select NP individuals for next generation
        std::vector<size_t> pop_next = selection(popnew, NP);
        for(population::size_type i = 0; i<NP; i++){
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
    auto intercepts = find_intercepts(R, ext_points);
    auto norm_objs = normalize_objectives(translated_objectives, intercepts);
    std::vector<ReferencePoint> rps = generate_uniform_reference_points(nobj, divisions);
    associate_with_reference_points(rps, norm_objs, fronts);

    // Apply RP selection to final front until N_pop reached
    while(next.size() < N_pop){
        size_t min_rp_idx = identify_niche_point(rps);
        std::optional<size_t> selected_idx = rps[min_rp_idx].select_member();
        if(selected_idx.has_value()){
            rps[min_rp_idx].increment_members();
            rps[min_rp_idx].remove_candidate(selected_idx.value());
            next.push_back(selected_idx.value());
        }else{
            rps.erase(rps.begin() + min_rp_idx);
        }
    }

    return next;
}

// Object serialization
template <typename Archive>
void nsga3::serialize(Archive &ar, unsigned int) {
    detail::archive(ar, ngen, cr, eta_c, mut, eta_mut, seed, m_verbosity, m_log);
}

}  // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::nsga3)
