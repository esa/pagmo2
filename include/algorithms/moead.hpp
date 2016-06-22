#ifndef PAGMO_ALGORITHMS_MOEAD_HPP
#define PAGMO_ALGORITHMS_MOEAD_HPP

#include <iomanip>
#include <random>
#include <string>
#include <tuple>

#include "../algorithm.hpp" // needed for the cereal macro
#include "../io.hpp"
#include "../exceptions.hpp"
#include "../population.hpp"
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
            bool preserve_diversity = true
            ) : m_gen(gen), m_weight_generation(weight_generation), m_T(T), m_CR(CR), m_F(F), m_eta_m(eta_m),
                m_realb(realb), m_limit(limit), m_preserve_diversity(preserve_diversity)
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
