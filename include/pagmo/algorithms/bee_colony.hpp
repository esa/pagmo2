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
    /// Single entry of the log (gen, fevals, curr_best, best, dx, df)
    typedef std::tuple<unsigned int, unsigned long long, double, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;
#else
    using log_line_type = std::tuple<unsigned int, unsigned long long, double, double, double, double>;
    using log_type = std::vector<log_line_type>;
#endif

    /// Constructor.
    /**
     * Constructs a bee_colony algorithm
     */
    bee_colony(unsigned int mfe = 1u, unsigned int limit = 1u, unsigned int seed = pagmo::random_device::next())
        : m_mfe(mfe), m_limit(limit), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
        if (limit == 0u) {
            pagmo_throw(std::invalid_argument, "The limit must be larger than 0.");
        }
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
        if (m_mfe == 0u) {
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
            0u, dim - 1u); // to generate a random index for the component to mutate
        std::uniform_int_distribution<vector_double::size_type> neirng(
            0u, NP - 2u); // to generate a random index for the neighbour

        auto gen = 1u;
        while (prob.get_fevals() - fevals0 < m_mfe) {
            // 1 - Employed bees phase
            for (decltype(NP) i = 0u; i < NP; ++i) {
                newsol = X[i];
                // selects a random component of the decision vector
                auto comp2change = comprng(m_e);
                // selects a random neighbour
                auto neighbour = neirng(m_e);
                if (neighbour >= i) {
                    ++neighbour;
                }
                // mutate new solution
                newsol[comp2change] += phirng(m_e) * (newsol[comp2change] - X[neighbour][comp2change]);
                // if the generated parameter value is out of boundaries, shift it into the boundaries
                if (newsol[comp2change] < lb[comp2change]) {
                    newsol[comp2change] = lb[comp2change];
                }
                if (newsol[comp2change] > ub[comp2change]) {
                    newsol[comp2change] = ub[comp2change];
                }
                // if the new solution is better than the old one replace it and reset its trial counter
                auto newfitness = prob.fitness(newsol);

                if (newfitness[0] < fit[i][0]) {
                    fit[i][0] = newfitness[0];
                    X[i][comp2change] = newsol[comp2change];
                    pop.set_xf(i, newsol, newfitness);
                    trial[i] = 0;
                } else {
                    ++trial[i];
                }
                if (prob.get_fevals() - fevals0 >= m_mfe) {
                    return pop;
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
                    ++t;
                    newsol = X[s];
                    // selects a random component of the decision vector
                    auto comp2change = comprng(m_e);
                    // selects a random neighbour
                    auto neighbour = neirng(m_e);
                    if (neighbour >= s) {
                        ++neighbour;
                    }
                    // mutate new solution
                    newsol[comp2change] += phirng(m_e) * (newsol[comp2change] - X[neighbour][comp2change]);
                    // if the generated parameter value is out of boundaries, shift it into the boundaries
                    if (newsol[comp2change] < lb[comp2change]) {
                        newsol[comp2change] = lb[comp2change];
                    }
                    if (newsol[comp2change] > ub[comp2change]) {
                        newsol[comp2change] = ub[comp2change];
                    }
                    // if the new solution is better than the old one replace it and reset its trial counter
                    auto newfitness = prob.fitness(newsol);
                    if (newfitness[0] < fit[s][0]) {
                        fit[s][0] = newfitness[0];
                        X[s][comp2change] = newsol[comp2change];
                        pop.set_xf(s, newsol, newfitness);
                        trial[s] = 0;
                    } else {
                        ++trial[s];
                    }
                    if (prob.get_fevals() - fevals0 >= m_mfe) {
                        return pop;
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
                if (prob.get_fevals() - fevals0 >= m_mfe) {
                    return pop;
                }
            }
            // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Every m_verbosity generations print a log line
                if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                    auto best_idx = pop.best_idx();
                    auto worst_idx = pop.worst_idx();
                    double dx = 0.;
                    // The population flattness in chromosome
                    for (decltype(dim) i = 0u; i < dim; ++i) {
                        dx += std::abs(pop.get_x()[worst_idx][i] - pop.get_x()[best_idx][i]);
                    }
                    // The population flattness in fitness
                    double df = std::abs(pop.get_f()[worst_idx][0] - pop.get_f()[best_idx][0]);
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "Current best:",
                              std::setw(15), "Best:", std::setw(15), "dx:", std::setw(15), "df:", '\n');
                    }
                    print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0, std::setw(15),
                          pop.get_f()[best_idx][0], std::setw(15), pop.champion_f()[0], std::setw(15), dx,
                          std::setw(15), df, '\n');
                    ++count;
                    // Logs
                    m_log.push_back(log_line_type(gen, prob.get_fevals() - fevals0, pop.get_f()[best_idx][0],
                                                  pop.champion_f()[0], dx, df));
                }
            }
            ++gen;
        }
        return pop;
    }

    /// Sets the seed
    /**
     * @param seed the seed controlling the algorithm stochastic behaviour
     */
    void set_seed(unsigned int seed)
    {
        m_e.seed(seed);
        m_seed = seed;
    }
    /// Gets the seed
    /**
     * @return the seed controlling the algorithm stochastic behaviour
     */
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
     * @code{.unparsed}
     *    Gen:        Fevals:  Current best:          Best:            dx:            df:
     *       1             50         203199         203199        63.3553    2.39397e+06
     *     101           5137         185.11         185.11        62.6044    2.50845e+06
     *     201          10237        36.8093        7.28722        55.2853    2.51614e+06
     *     301          15337        45.2122        7.28722        52.4086    2.18182e+06
     *     401          20437        18.3259        7.28722        36.9614    2.67986e+06
     *     501          25537         261364        7.28722        58.7791    2.69368e+06
     * @endcode
     * Gen is the generation number, Fevals the number of function evaluation used, Current best is the best fitness
     * currently in the population, Best is the best fitness found, dx is the population flatness evaluated as
     * the distance between the decisions vector of the best and of the worst individual, df is the population flatness
     * evaluated as the distance between the fitness of the best and of the worst individual.
     *
     * @param level verbosity level
     */
    void set_verbosity(unsigned int level)
    {
        m_verbosity = level;
    }
    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned int get_verbosity() const
    {
        return m_verbosity;
    }
    /// Gets the number of maximum function evaluations
    /**
     * @return the number of maximum function evaluations
     */
    unsigned int get_mfe() const
    {
        return m_mfe;
    }
    /// Algorithm name
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing the algorithm name
     */
    std::string get_name() const
    {
        return "Artificial Bee Colony";
    }
    /// Extra informations
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tMaximum number of objective function evaluations: ", m_mfe);
        stream(ss, "\n\tLimit: ", m_limit);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        stream(ss, "\n\tSeed: ", m_seed);
        return ss.str();
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
        ar(m_mfe, m_limit, m_e, m_seed, m_verbosity, m_log);
    }

private:
    unsigned int m_mfe;
    unsigned int m_limit;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::bee_colony)

#endif
