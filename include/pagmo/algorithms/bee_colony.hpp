#ifndef PAGMO_ALGORITHMS_BEE_COLONY_HPP
#define PAGMO_ALGORITHMS_BEE_COLONY_HPP

#include <random>
#include <string>

#include "../algorithm.hpp"
#include "../exceptions.hpp"
#include "../io.hpp"
#include "../population.hpp"
#include "../rng.hpp"
#include "../utils/generic.hpp"

namespace pagmo
{
/// Artificial Bee Colony Algorithm
/**
 *
 */
class bee_colony
{
public:
#if defined(DOXYGEN_INVOKED)
    /// Single entry of the log (gen, fevals, best, dx, df)
    typedef std::tuple<unsigned int, unsigned long long, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;
#else
    using log_line_type = std::tuple<unsigned int, unsigned long long, double, double, double>;
    using log_type = std::vector<log_line_type>;
#endif

    /// Constructor.
    /**
     * Constructs a bee_colony algorithm
     */
    bee_colony(unsigned int gen = 1u, unsigned int limit = 1u, unsigned int seed = pagmo::random_device::next())
        : m_gen(gen), m_limit(limit), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
    }

    /// Algorithm evolve method
    /**
     * Evolves the population
     */
    population evolve(population pop) const
    {
        const auto &prob = pop.get_problem();
        auto dim = prob.get_nx();
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto NP = pop.size();
        auto prob_f_dimension = prob.get_nf();
        auto fevals0 = prob.get_fevals(); // disount for the already made fevals
        auto count = 1u;                  // regulates the screen output
        // PREAMBLE-------------------------------------------------------------------------------------------------
        // Check whether the problem/population are suitable for bee_colony
        if (prob_f_dimension != 1u) {
            pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The problem appears to be stochastic. " + get_name() + " cannot deal with it");
        }
        if (NP < 2u) {
            pagmo_throw(std::invalid_argument, prob.get_name() + " needs at least 2 individuals in the population, "
                                                   + std::to_string(NP) + " detected");
        }
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Some vectors used during evolution are declared.
        vector_double newsol(dim); // contains the mutated candidate
        auto X = pop.get_x();
        auto fit = pop.get_f();
        std::vector<unsigned> trial(NP, 0u);
        std::uniform_real_distribution<double> phirng(-1., 1.); // to generate a number in [-1, 1]
        std::uniform_real_distribution<double> rrng(0., 1.);    // to generate a number in [0, 1]
        std::uniform_int_distribution<vector_double::size_type> comprng(
            0u, dim - 1u); // to generate a random index for the component
        std::uniform_int_distribution<vector_double::size_type> neirng(
            0u, NP - 2u); // to generate a random index for the neighbour

        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
            // 1 - Employed bees phase
            for (decltype(NP) i = 0u; i < NP; ++i) {
                newsol = X[i];
                // selects a random component of the decision vector
                auto comp2change = comprng(m_e);
                // selects a random neighbour
                auto neighbour = neirng(m_e);
                if (neighbour >= i) {
                    neighbour++;
                }
                // mutate new solution
                newsol[comp2change] += phirng(m_e) * (newsol[comp2change] - X[neighbour][comp2change]);
                // if the generated parameter value is out of boundaries, shift it into the boundaries
                if (newsol[comp2change] < lb[comp2change]) {
                    newsol[comp2change] = lb[comp2change]
                }
                if (newsol[comp2change] > ub[comp2change]) {
                    newsol[comp2change] = ub[comp2change]
                }
                // if the new solution is better than the old one replace it and reset its trial counter
                auto newfitness = prob.fitness(newsol);
                if (newfitness < fit[i][0]) {
                    fit[i][0] = newfitness;
                    X[i][comp2change] = newsol[comp2change];
                    pop.set_xf(i, newsol, newfitness);
                    trial[i] = 0;
                } else {
                    trial[i]++;
                }
            }

            // 2 - Onlooker bee phase
            // compute probabilities
            vector_double p(NP);
            auto sump = 0.;
            for (decltype(NP) i = 0u; i < NP; ++i) {
                if (fit[i][0] >= 0.) {
                    p[i] = 1. / (1. + fit[i][0]);
                } else {
                    p[i] = 1. - fit[i][0];
                }
                sump += p[i];
            }
            for (decltype(NP) i = 0u; i < NP; ++i) {
                p[i] /= sump;
            }
            auto s = 0u;
            auto t = 0u;
            // for each onlooker bee
            while (t < NP) {
                // probabilistic selection of a food source
                auto r = rrng(m_e);
                if (r < p[s]) {
                    t++;
                    newsol = X[s];
                    // selects a random component of the decision vector
                    auto comp2change = comprng(m_e);
                    // selects a random neighbour
                    auto neighbour = neirng(m_e);
                    if (neighbour >= s) {
                        neighbour++;
                    }
                    // mutate new solution
                    newsol[comp2change] += phirng(m_e) * (newsol[comp2change] - X[neighbour][comp2change]);
                    // if the generated parameter value is out of boundaries, shift it into the boundaries
                    if (newsol[comp2change] < lb[comp2change]) {
                        newsol[comp2change] = lb[comp2change]
                    }
                    if (newsol[comp2change] > ub[comp2change]) {
                        newsol[comp2change] = ub[comp2change]
                    }
                    // if the new solution is better than the old one replace it and reset its trial counter
                    auto newfitness = prob.fitness(newsol);
                    if (newfitness < fit[s][0]) {
                        fit[s][0] = newfitness;
                        X[s][comp2change] = newsol[comp2change];
                        pop.set_xf(s, newsol, newfitness);
                        trial[s] = 0;
                    } else {
                        trial[s]++;
                    }
                }
                s = (s + 1) % NP;
            }
            // 3 - Scout bee phase
            auto mi = 0u;
            for (auto i = 1u; i < NP; ++i) {
                if (trial[i] > trial[mi]) {
                    mi = i;
                }
            }
            if (trial[mi] > m_limit) {
                for (auto j = 0u; j < dim; ++j) {
                    X[mi][j] = uniform_real_from_range(lb[j], ub[j], m_e);
                }
                auto newfitness = prob.fitness(X[mi]);
                pop.set_xf(mi, X[mi], newfitness);
                trial[mi] = 0;
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
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >0: will print and log one line each \p level generations.
     *
     * Example (verbosity 100):
     * @code
     * @endcode
     *
     * @param level verbosity level
     */
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
        return "Artificial Bee Colony";
    }
    /// Extra informations
    std::string get_extra_info() const
    {
        return "\tGenerations: " + std::to_string(m_gen) + "\n\tVerbosity: " + std::to_string(m_verbosity)
               + "\n\tSeed: " + std::to_string(m_seed);
    }
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve.
     */
    const log_type &get_log() const
    {
        return m_log;
    }
    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_gen, m_e, m_seed, m_verbosity, m_log);
    }

private:
    unsigned int m_gen;
    unsigned int m_limit;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::bee_colony)

#endif
