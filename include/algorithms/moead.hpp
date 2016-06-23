#ifndef PAGMO_ALGORITHMS_MOEAD_HPP
#define PAGMO_ALGORITHMS_MOEAD_HPP

#include <algorithm> // std::shuffle, std::transform
#include <cmath>     // std::tgamma
#include <iomanip>
#include <numeric>   // std::iota, std::inner_product
#include <random>
#include <string>
#include <tuple>

#include "../algorithm.hpp" // needed for the cereal macro
#include "../io.hpp"
#include "../exceptions.hpp"
#include "../population.hpp"
#include "../problems/decompose.hpp"
#include "../utils/generic.hpp" // safe_cast
#include "../utils/multi_objective.hpp" // ideal
#include "../utils/discrepancy.hpp" // halton
#include "../rng.hpp"

namespace pagmo
{

class moead
{
public:
    /// Constructor
     /**
     * Constructs a MOEA/D-DE algorithm
     *
     * @param[in] gen number of generations
     * @param[in] weight_generation method used to generate the weights, one of "grid", "low discrepancy" or "random")
     * @param[in] T size of the weight's neighborhood
     * @param[in] CR Crossover parameter in the Differential Evolution operator
     * @param[in] F parameter for the Differential Evolution operator
     * @param[in] eta_m Distribution index used by the polynomial mutation
     * @param[in] realb chance that a neighbourhood of dimension T is considered at each generation for each weight, rather than the whole population (only if preserve_diversity is true)
     * @param[in] limit Maximum number of copies reinserted in the population  (only if m_preserve_diversity is true)
     * @param[in] preserve_diversity when true activates the two diversity preservation mechanisms described in Li, Hui, and Qingfu Zhang paper
     * @throws value_error if gen is negative, weight_generation is not one of the allowed types, realb,cr or f are not in [1.0] or m_eta is < 0
     */
    moead(unsigned int gen = 1u,
            std::string weight_generation = "grid",
            population::size_type T = 20u,
            double CR = 1.0,
            double F = 0.5,
            double eta_m = 20.,
            double realb = 0.9,
            unsigned int limit = 2u,
            bool preserve_diversity = true,
            unsigned int seed = pagmo::random_device::next()
            ) : m_gen(gen), m_weight_generation(weight_generation), m_T(T), m_CR(CR), m_F(F), m_eta_m(eta_m),
                m_realb(realb), m_limit(limit), m_preserve_diversity(preserve_diversity), m_e(seed), m_seed(seed), m_verbosity(0u)
    {
        // Sanity checks
        if(m_weight_generation != "random" && m_weight_generation != "grid" && m_weight_generation != "low discrepancy") {
            pagmo_throw(std::invalid_argument, "Weight generation method requested is '" + m_weight_generation + "', but only one of 'random', 'low discrepancy', 'grid' is allowed");
        }
        if(CR > 1.0 || CR < 0.) {
            pagmo_throw(std::invalid_argument, "The parameter CR (used by the differential evolution operator) needs to be in [0,1], while a value of " + std::to_string(CR) + " was detected");
        }
        if(F > 1.0 || F < 0.) {
            pagmo_throw(std::invalid_argument, "The parameter F (used by the differential evolution operator) needs to be in [0,1], while a value of " + std::to_string(F) + " was detected");
        }
        if(eta_m < 0.) {
            pagmo_throw(std::invalid_argument, "The distribution index for the polynomial mutation (eta_m) needs to be positive, while a value of " + std::to_string(eta_m) + " was detected");
        }
        if(realb > 1.0 || realb < 0.) {
            pagmo_throw(std::invalid_argument, "The chance of considering a neighbourhood (realb) needs to be in [0,1], while a value of " + std::to_string(realb) + " was detected");
        }
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     *
     * Evolves the population for the requested number of generations.
     *
     * @param[in] pop population to be evolved
     * @return evolved population
     */
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem();       // This is a const reference, so using set_seed for example will not be allowed (pop.set_problem_seed is)
        auto dim = prob.get_nx();                   // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto NP = pop.size();

        //auto fevals0 = prob.get_fevals();           // disount for the already made fevals
        //unsigned int count = 1u;                    // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this
        // particular algorithm.
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument,"Non linear constraints detected in " + prob.get_name() + " instance. " + get_name() + " cannot deal with them");
        }
        if (prob.get_nf() < 2u) {
            pagmo_throw(std::invalid_argument,"Number of objectives detected in " + prob.get_name() + " instance is " + std::to_string(prob.get_nf()) + ". " + get_name() + " necessitates a problem with multiple objectives");
        }
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,"The problem appears to be stochastic " + get_name() + " cannot deal with it");
        }
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        if ( m_T > NP - 1u ) {
            pagmo_throw(std::invalid_argument, "The neighbourhood size specified (T) is " + std::to_string(m_T) + ": too large for the input population having size " + std::to_string(NP) );
        }
        // Generate the vector of weight's vectors for the NP decomposed problems. Will throw if the population size is not compatible with the weight generation scheme chosen
        auto weights = generate_weights(prob.get_nf(), NP);
        print("Weights: ", weights, '\n');
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        //m_log.clear();



        // Setting up necessary quantities------------------------------------------------------------------------------
        // Random distributions
        std::uniform_real_distribution<double> drng(0.,1.); // to generate a number in [0, 1)
        std::uniform_int_distribution<vector_double::size_type> p_idx(0u, NP - 1u); // to generate a random index for the population
        // Declaring the candidate chromosome
    	vector_double candidate(dim);
        // We compute, for each vector of weights, the k = m_T neighbours
        auto neigh_idxs = compute_neighbours(weights);
        print("Neighbours ids: ", neigh_idxs, '\n');
        // We compute the initial ideal point (will be adapted along the course of the algorithm)
        vector_double ideal_point = ideal(pop.get_f());
        // We create a decompose problem which will be used only to access its decompose_fitness(f) method
        // (the construction parameter weights, ideal_point and false are thus irrelevant).
        decompose prob_decomposed{prob, weights[0], ideal_point, "tchebycheff", false};
        // We create the container that will represent a pseudo-random permutation of the population indexes 1..NP
        std::vector<population::size_type> shuffle(NP);
        std::iota(shuffle.begin(), shuffle.end(), std::vector<population::size_type>::size_type(0u));

        // Main MOEA/D loop --------------------------------------------------------------------------------------------
        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
        // 1 - Shuffle the population indexes
        std::shuffle(shuffle.begin(), shuffle.end(), m_e);
        // 2 - Loop over the shuffled NP decomposed problems
            for (auto n : shuffle) {
                // 3 - if the diversity preservation mechanism is active we select at random whether to consider the whole
                // population or just a neighbourhood to select two parents
                bool whole_population;
                if(drng(m_e) < m_realb || !m_preserve_diversity) {
                    whole_population = false;	// neighborhood
                }
                else {
                    whole_population = true;	// whole population
                }
                // 4 - We select two parents in the neighbourhood
                std::vector<population::size_type> parents_idx(2);
                parents_idx = select_parents(n, neigh_idxs, whole_population);
                // 5 - Crossover using the Differential Evolution operator (binomial crossover)
                for(decltype(dim) kk = 0u; kk < dim; ++kk)
                {
                    if (drng(m_e) < m_CR) {
                        /*Selected Two Parents*/
                        candidate[kk] = pop.get_x()[n][kk] + m_F * (pop.get_x()[parents_idx[0]][kk] - pop.get_x()[parents_idx[1]][kk]);
                        // Fix the bounds
                        if(candidate[kk] < lb[kk]){
                            candidate[kk] = lb[kk] + drng(m_e)*(pop.get_x()[n][kk] - lb[kk]);
                        }
                        if(candidate[kk] > ub[kk]){
                            candidate[kk] = ub[kk] - drng(m_e) * (ub[kk] - pop.get_x()[n][kk]);
                        }
                    } else {
                        candidate[kk] = pop.get_x()[n][kk];
                    }
                }
                // 6 - We apply a further mutation using polynomial mutation
                polynomial_mutation(candidate, pop, 1.0 / static_cast<double>(dim));
                // 7- We evaluate the fitness function.
                auto new_f = prob.fitness(candidate);
                // 8 - We update the ideal point
                for (decltype(prob.get_nf()) j = 0u; j < prob.get_nf(); ++j) {
                    ideal_point[j] = std::min(new_f[j], ideal_point[j]);
                }
                std::transform(ideal_point.begin(), ideal_point.end(), new_f.begin(), ideal_point.begin(), [](double a, double b){return std::min(a,b);});
                // 9 - We insert the newly found solution into the population
                decltype(NP) size, time = 0;
                // First try on problem n
                auto f1 = prob_decomposed.decompose_fitness(pop.get_f()[n], weights[n], ideal_point);
                auto f2 = prob_decomposed.decompose_fitness(new_f, weights[n], ideal_point);
                if(f2[0]<f1[0])
                {
                    pop.set_xf(n, candidate, new_f);
                    time++;
                }
                // Then, on neighbouring problems up to m_limit (to preserve diversity)
                if (whole_population) {
                    size = NP;
                } else {
                    size=neigh_idxs[n].size();
                }
                std::vector<population::size_type> shuffle2(size);
                std::iota(shuffle2.begin(), shuffle2.end(), std::vector<population::size_type>::size_type(0u));
                std::shuffle(shuffle2.begin(), shuffle2.end(), m_e);
                for (decltype(size) k = 0u; k < size; ++k) {
                    population::size_type pick;
                    if (whole_population) {
                        pick = shuffle2[k];
                    } else {
                        pick = neigh_idxs[n][shuffle2[k]];
                    }
                    f1 = prob_decomposed.decompose_fitness(pop.get_f()[pick], weights[pick], ideal_point);
                    f2 = prob_decomposed.decompose_fitness(new_f, weights[pick], ideal_point);
                    if(f2[0] < f1[0])
                    {
                        pop.set_xf(pick, candidate, new_f);
                        time++;
                    }
                    // the maximal number of solutions updated is not allowed to exceed 'limit' if diversity is to be preserved
                    if(time >= m_limit && m_preserve_diversity) {
                        break;
                    }
                }
            }
        }
        return pop;
    }

    /// Sets the algorithm seed
    void set_seed(unsigned int seed)
    {
        m_seed = seed;
    };
    /// Gets the seed
    unsigned int get_seed() const
    {
        return m_seed;
    }
    /// Sets the algorithm verbosity
    void set_verbosity(unsigned int level)
    {
        m_verbosity = level;
    };
    /// Gets the verbosity level
    unsigned int get_verbosity() const
    {
        return m_verbosity;
    }
    /// Get generations
    unsigned int get_gen() const
    {
        return m_gen;
    }
    /// Algorithm name
    std::string get_name() const
    {
        return "MOEA/D - DE";
    }
    /// Extra informations
    std::string get_extra_info() const
    {
        return "\tGenerations: " + std::to_string(m_gen) +
            "\n\tWeight generation: " + m_weight_generation +
            "\n\tNeighbourhood size: " + std::to_string(m_T) +
            "\n\tParameter CR: " + std::to_string(m_CR) +
            "\n\tParameter F: " + std::to_string(m_F) +
            "\n\tDistribution index: " + std::to_string(m_eta_m) +
            "\n\tChance for diversity preservation: " + std::to_string(m_realb) +
            "\n\tVerbosity: " + std::to_string(m_verbosity) +
            "\n\tSeed: " + std::to_string(m_seed);
    }
    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_gen, m_weight_generation, m_T, m_CR, m_F, m_eta_m, m_realb, m_limit, m_preserve_diversity, m_e, m_seed, m_verbosity);
    }
private:
    //Recursive function building all m-ple of elements of X summing to s
    void reksum(
            std::vector<std::vector<double> > &retval,
            const std::vector<population::size_type>& X,
            population::size_type m,
            population::size_type s,
            std::vector<double> eggs = std::vector<double>() ) const
    {
        if (m==1u) {
            if (std::find(X.begin(),X.end(), s) == X.end()) { //not found
                return;
            } else {
                eggs.push_back(static_cast<double>(s));
                retval.push_back(eggs);
            }
        } else {
            for (decltype(X.size()) i = 0u; i < X.size(); ++i) {
                eggs.push_back(static_cast<double>(X[i]));
                reksum(retval, X , m - 1u, s - X[i], eggs);
                eggs.pop_back();
            }
        }
    }

    // Binomial coefficient implemented via gamma functions
    double binomial_coefficient(vector_double::size_type n, vector_double::size_type k) const
    {
        return std::exp(std::lgamma(static_cast<double>(n) + 1.) - std::lgamma(static_cast<double>(k) + 1.) - std::lgamma(static_cast<double>(n) - static_cast<double>(k) + 1.));
    }

    /// Weights generation
    /**
     * Generates the weight vectors used to decompose the problem.
     *
     * @param[in] n_f dimension of each weight vector (i.e. fitness dimension)
     * @param[in] n_w number of weights (i.e. population size)
     *
     * @returns an <tt>std:vector<\tt> containing the weight vectors
     *
     * @throws if the population size is not compatible with the selected weight generation method
    **/
     std::vector<vector_double> generate_weights(vector_double::size_type n_f, vector_double::size_type n_w) const
    {
        // Sanity check
        if (n_f > n_w) {
             pagmo_throw(std::invalid_argument,"A fitness size of " + std::to_string(n_f) + " was requested to the weight generation routine, while " + std::to_string(n_w) + " weights were requested to be generated. To allow weight be generated correctly the number of weights must be strictly larger than the number of objectives");
        }

        // Random distributions
        std::uniform_real_distribution<double> drng(0.,1.); // to generate a number in [0, 1)
        std::vector<vector_double> retval;
        if(m_weight_generation == "grid") {
            // find the largest H resulting in a population smaller or equal to NP
            decltype(n_w) H;
            if (n_f == 2u) {
                H = n_w - 1u;
            } else if (n_f == 3u) {
                H = static_cast<decltype(H)>(std::floor(0.5 * (std::sqrt(8. * static_cast<double>(n_w) + 1.) - 3.)));
            } else {
                H = 1u;
                while(binomial_coefficient(H + n_f - 1u, n_f - 1u) <= static_cast<double>(n_w))
                {
                    ++H;
                }
                H--;
            }

            // We check that NP equals the population size resulting from H
            if (std::abs(static_cast<double>(n_w) - binomial_coefficient(H + n_f - 1u, n_f - 1u)) > 1E-8 ) {
                std::ostringstream error_message;
                error_message << "Population size of " << std::to_string(n_w) << " is detected, but not supported by the '" << m_weight_generation
                    << "' weight generation method selected for " << get_name() << ". A size of " << binomial_coefficient(H + n_f - 1u, n_f - 1u)
                    << " or " << binomial_coefficient(H + n_f, n_f - 1u)
                    << " is possible.";
                    pagmo_throw(std::invalid_argument, error_message.str());
            }

            // We generate the weights
            std::vector<population::size_type> range(H + 1u);
            std::iota(range.begin(), range.end(), std::vector<population::size_type>::size_type(0u));
            reksum(retval, range, n_f, H);
            for(decltype(retval.size()) i = 0u; i < retval.size(); ++i) {
                for(decltype(retval[i].size()) j = 0u; j < retval[i].size(); ++j) {
                    retval[i][j] /= static_cast<double>(H);
                }
            }
        } else if(m_weight_generation == "low discrepancy") {
            // We first push back the "corners" [1,0,0,...], [0,1,0,...]
            for(decltype(n_f) i = 0u; i < n_f; ++i) {
                retval.push_back(vector_double(n_f, 0.));
                retval[i][i] = 1.;
            }
            // Then we add points on the simplex randomly genrated using Halton low discrepancy sequence
            halton ld_seq{safe_cast<unsigned int>(n_f), 1u};
            for(decltype(n_w) i = n_f; i < n_w; ++i) {
                retval.push_back(sample_from_simplex(ld_seq()));
            }
        } else if(m_weight_generation == "random") {
            for (decltype(n_w) i = 0u; i < n_w; ++i) {
                vector_double dummy(n_f - 1u, 0.);
                for(decltype(n_f) j = 0u; j < n_f - 1u; ++j) {
                    dummy[j] = drng(m_e);
                }
                retval.push_back(sample_from_simplex(dummy));
            }
        }
        return retval;
     }

     // Performs polynomial mutation (same as nsgaII)
     void polynomial_mutation(vector_double& child, const population& pop, double rate) const
     {
        const auto &prob = pop.get_problem();
        auto D = prob.get_nx();                   // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        double rnd, delta1, delta2, mut_pow, deltaq;
        double y, yl, yu, val, xy;
        std::uniform_real_distribution<double> drng(0.,1.); // to generate a number in [0, 1)

        //This implements the real polinomial mutation of an individual
        for (decltype(D) j = 0u; j < D; ++j){
            if (drng(m_e) <= rate) {
                y = child[j];
                yl = lb[j];
                yu = ub[j];
                delta1 = (y-yl)/(yu-yl);
                delta2 = (yu-y)/(yu-yl);
                rnd = drng(m_e);
                mut_pow = 1. / (m_eta_m + 1.);
                if (rnd <= 0.5)
                {
                    xy = 1. - delta1;
                    val = 2. * rnd+(1. - 2.*rnd)*(std::pow(xy,(m_eta_m + 1.)));
                    deltaq =  std::pow(val,mut_pow) - 1.;
                }
                else
                {
                    xy = 1. - delta2;
                    val = 2. * (1. - rnd) + 2. * (rnd - 0.5) * (std::pow(xy,(m_eta_m + 1.)));
                    deltaq = 1. - (std::pow(val,mut_pow));
                }
                y = y + deltaq*(yu-yl);
                if (y<yl) y = yl;
                if (y>yu) y = yu;
                child[j] = y;
            }
        }
     }

    std::vector<population::size_type> select_parents(population::size_type n, const std::vector<std::vector<population::size_type> >& neigh_idx, bool whole_population) const
    {
        std::vector<population::size_type> retval;
        auto ss = neigh_idx[n].size();
        decltype(ss) p;

        std::uniform_int_distribution<vector_double::size_type> p_idx(0, neigh_idx.size() - 1u); // to generate a random index for the neighbourhood

        while(retval.size() < 2u)
        {
            if(!whole_population){
                p = neigh_idx[n][p_idx(m_e) % ss];
            } else {
                p = p_idx(m_e);
            }
            bool flag = true;
            for(decltype(retval.size()) i = 0u; i < retval.size(); i++)
            {
                if(retval[i] == p) // p is in the list
                {
                    flag = false;
                    break;
                }
            }
            if(flag) retval.push_back(p);
        }
        return retval;
    }

    // The returned value [i][j] component will contain the j-th closest vector
    // (according to the euclidian distance) to the i-th vector. with j=1..m_T
    std::vector<std::vector<population::size_type> > compute_neighbours(const std::vector<vector_double> &weights) const {
        std::vector<std::vector<population::size_type> > neigh_idxs;
        // loop through the weights
        for(decltype(weights.size()) i = 0u; i < weights.size(); ++i) {
            // We compute all the distances to all other weights including the self
            vector_double distances;
            for(decltype(weights.size()) j = 0u; j < weights.size(); ++j) {
                double dist = 0.;
                for (decltype(weights[i].size()) k = 0u; k < weights[i].size(); ++k) {
                    dist += (weights[i][k] - weights[j][k]) * (weights[i][k] - weights[j][k]);
                }
                distances.push_back(std::sqrt(dist));
            }
            // We sort the indexes with respect to the distance
            std::vector<population::size_type> idxs(weights.size());
            std::iota(idxs.begin(), idxs.end(), population::size_type(0u));
            std::sort(idxs.begin(), idxs.end(), [&distances] (auto idx1, auto idx2) {return distances[idx1] < distances[idx2];});
print("distances: ", distances);
            neigh_idxs.push_back(idxs);
        }
        // We remove the first element containg the self-distance (0) and crop the rest to m_T
        for (decltype(neigh_idxs.size()) i = 0u; i < neigh_idxs.size();++i) {
            neigh_idxs[i].erase(neigh_idxs[i].begin());
            neigh_idxs[i].erase(neigh_idxs[i].begin() + m_T, neigh_idxs[i].end());
        }
        return neigh_idxs;
    }

    unsigned int                        m_gen;
    std::string                         m_weight_generation;
    population::size_type               m_T;
    double                              m_CR;
    double                              m_F;
    double                              m_eta_m;
    double                              m_realb;
    unsigned int                        m_limit;
    bool                                m_preserve_diversity;
    mutable detail::random_engine_type  m_e;
    unsigned int                        m_seed;
    unsigned int                        m_verbosity;
};

} //namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::moead)

#endif
