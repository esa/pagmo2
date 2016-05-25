#ifndef PAGMO_ALGORITHMS_DE_HPP
#define PAGMO_ALGORITHMS_DE_HPP

#include <iomanip>
#include <numeric> //std::iota
#include <random>
#include <utility> //std::swap
#include <string>
#include <tuple>

#include "../io.hpp"
#include "../exceptions.hpp"
#include "../population.hpp"
#include "../rng.hpp"
#include "../utils/generic.hpp"

namespace pagmo
{
class de
{
public:
    #if defined(DOXYGEN_INVOKED)
        /// Single entry of the log
        typedef std::tuple<unsigned int, unsigned long long, double, double, double> log_line_type;
        /// The log
        typedef std::vector<log_line_type> log_type;
    #else
        using log_line_type = std::tuple<unsigned int, unsigned long long, double, double, double>;
        using log_type = std::vector<log_line_type>;
    #endif

    de(unsigned int gen = 1u, double F = 0.7, double CR = 0.5, unsigned int strategy = 2u, double ftol = 1e-6, double xtol = 1e-6, unsigned int seed = pagmo::random_device::next()) :
        m_gen(gen), m_F(F), m_CR(CR), m_strategy(strategy), m_ftol(ftol), m_xtol(xtol), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
        if (strategy < 1u || strategy > 10u) {
            pagmo_throw(std::invalid_argument, "The Differential Evolution strategy must be in [1, .., 10], while a value of " + std::to_string(strategy) + " was detected.");
        }
        if (CR < 0. || F < 0. || CR > 1. || F > 1.) {
            pagmo_throw(std::invalid_argument, "The F and CR parameters must be in the [0,1] range");
        }
    }

    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem();       // This is a const reference, so using set_seed for example will not be allowed
        auto dim = prob.get_nx();                   // This getter does not return a const reference but a copy
        const auto &bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto NP = pop.size();
        auto prob_f_dimension = prob.get_nf();
        auto fevals0 = prob.get_fevals();           // disount for the already made fevals
        unsigned int count = 1u;                    // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this
        // particular algorithm.
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument,"Non linear constraints detected in " + prob.get_name() + " instance. " + get_name() + " cannot deal with them");
        }
        if (prob_f_dimension != 1u) {
            pagmo_throw(std::invalid_argument,"Multiple objectives detected in " + prob.get_name() + " instance. " + get_name() + " cannot deal with them");
        }
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        // ---------------------------------------------------------------------------------------------------------

        // We add some checks that are algorithm specific
        //
        if (NP < 6) {
            pagmo_throw(std::invalid_argument, prob.get_name() + " needs at least 6 individuals in the population, " + std::to_string(NP) + " detected");
        }

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Some vectors used during evolution are declared here.
        vector_double tmp(dim);                             // contains the mutated candidate
        std::uniform_real_distribution<double> drng(0.,1.); // to generate a number in [0, 1)]
        std::uniform_int_distribution<vector_double::size_type> rand_ind_idx(0u, dim - 1u); // to generate a random index in pop

        // We extract from pop the chromosomes and fitness associated
        auto popold = pop.get_x();
        auto fit = pop.get_f();
        auto popnew = popold;

        // Initialise the global bests
        auto best_idx = pop.best_idx();
        vector_double::size_type worst_idx = 0u;
        auto gbX = popnew[best_idx];
        auto gbfit=fit[best_idx];
        // the best decision vector of a generation
        auto gbIter = gbX;
        std::vector<vector_double::size_type> r(5);   //indexes of 5 selected population members

        // Main DE iterations
        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
            //Start of the loop through the deme
            for (decltype(NP) i = 0u; i < NP; ++i) {
                /*-----We select at random 5 indexes from the population---------------------------------*/
                std::vector<vector_double::size_type> idxs(NP);
                std::iota(idxs.begin(), idxs.end(), 0u);
                for (auto j = 0u; j < 5u; ++j) { // Durstenfeld's algorithm to select 5 indexes at random
                    auto idx = std::uniform_int_distribution<vector_double::size_type>(0u, NP - 1u - j)(m_e);
                    r[j] = idxs[idx];
                    std::swap(idxs[idx], idxs[NP - 1u - j]);
                }


                /*-------DE/best/1/exp--------------------------------------------------------------------*/
                /*-------The oldest DE strategy but still not bad. However, we have found several---------*/
                /*-------optimization problems where misconvergence occurs.-------------------------------*/
                if (m_strategy == 1u) { /* strategy DE0 (not in the original paper on DE) */
                    tmp = popold[i];
                    auto n = rand_ind_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = gbIter[n] + m_F * (popold[r[1]][n] - popold[r[2]][n]);
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < m_CR) && (L < dim));
                }

                /*-------DE/rand/1/exp-------------------------------------------------------------------*/
                /*-------This is one of my favourite strategies. It works especially well when the-------*/
                /*-------"gbIter[]"-schemes experience misconvergence. Try e.g. m_f=0.7 and m_cr=0.5---------*/
                /*-------as a first guess.---------------------------------------------------------------*/
                else if (m_strategy == 2u) { /* strategy DE1 */
                    tmp = popold[i];
                    auto n = rand_ind_idx(m_e);
                    decltype(dim) L = 0u;
                    do {
                        tmp[n] = popold[r[0]][n] + m_F * (popold[r[1]][n] - popold[r[2]][n]);
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < m_CR) && (L < dim));
                }
                /*-------DE/rand-to-best/1/exp-----------------------------------------------------------*/
                /*-------This strategy seems to be one of the best strategies. Try m_f=0.85 and m_cr=1.------*/
                /*-------If you get misconvergence try to increase NP. If this doesn't help you----------*/
                /*-------should play around with all three control variables.----------------------------*/
                else if (m_strategy == 3u) { /* similiar to DE2 but generally better */
                    tmp = popold[i];
                    auto n = rand_ind_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = tmp[n] + m_F * (gbIter[n] - tmp[n]) + m_F * (popold[r[0]][n] - popold[r[1]][n]);
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < m_CR) && (L < dim));
                }
                /*-------DE/best/2/exp is another powerful strategy worth trying--------------------------*/
                else if (m_strategy == 4u) {
                    tmp = popold[i];
                    auto n = rand_ind_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = gbIter[n] +
                        (popold[r[0]][n] + popold[r[1]][n] - popold[r[2]][n] - popold[r[3]][n]) * m_F;
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < m_CR) && (L < dim));
                }
                /*-------DE/rand/2/exp seems to be a robust optimizer for many functions-------------------*/
                else if (m_strategy == 5u) {
                    tmp = popold[i];
                    auto n = rand_ind_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = popold[r[4]][n] +
                            (popold[r[0]][n]+popold[r[1]][n]-popold[r[2]][n]-popold[r[3]][n]) * m_F;
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < m_CR) && (L < dim));
                }

                /*=======Essentially same strategies but BINOMIAL CROSSOVER===============================*/
                /*-------DE/best/1/bin--------------------------------------------------------------------*/
                else if (m_strategy == 6u) {
                    tmp = popold[i];
                    auto n = rand_ind_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) { /* perform Dc binomial trials */
                        if ((drng(m_e) < m_CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = gbIter[n] + m_F * (popold[r[1]][n] - popold[r[2]][n]);
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand/1/bin-------------------------------------------------------------------*/
                else if (m_strategy == 7u) {
                    tmp = popold[i];
                    auto n = rand_ind_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) { /* perform Dc binomial trials */
                        if ((drng(m_e) < m_CR) || L + 1u == dim) { /* change at least one parameter */
                        tmp[n] = popold[r[0]][n] + m_F * (popold[r[1]][n] - popold[r[2]][n]);
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand-to-best/1/bin-----------------------------------------------------------*/
                else if (m_strategy == 8u) {
                    tmp = popold[i];
                    auto n = rand_ind_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) { /* perform Dc binomial trials */
                        if ((drng(m_e) < m_CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = tmp[n] + m_F * (gbIter[n] - tmp[n]) + m_F * (popold[r[0]][n] - popold[r[1]][n]);
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/best/2/bin--------------------------------------------------------------------*/
                else if (m_strategy == 9u) {
                    tmp = popold[i];
                    auto n = rand_ind_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) { /* perform Dc binomial trials */
                        if ((drng(m_e) < m_CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = gbIter[n] +
                                (popold[r[0]][n] + popold[r[1]][n] - popold[r[2]][n] - popold[r[3]][n]) * m_F;
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand/2/bin--------------------------------------------------------------------*/
                else if (m_strategy == 10u) {
                    tmp = popold[i];
                    auto n = rand_ind_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) { /* perform Dc binomial trials */
                        if ((drng(m_e) < m_CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = popold[r[4]][n] +
                                (popold[r[0]][n] + popold[r[1]][n] - popold[r[2]][n] - popold[r[3]][n]) * m_F;
                        }
                        n = (n + 1u) % dim;
                    }
                }

                /*==Trial mutation now in tmp. force feasibility and see how good this choice really was.==*/
                // a) feasibility
                for (decltype(dim) j = 0u; j < dim; ++j) {
                    if ((tmp[j] < lb[j]) || (tmp[j] > ub[j])) {
                        tmp[j] = uniform_real_from_range(lb[j], ub[j], m_e);
                    }
                }
                //b) how good?
                auto newfitness = prob.fitness(tmp);        /* Evaluates tmp[] */
                if ( newfitness[0] <= fit[i][0] ) {         /* improved objective function value ? */
                    fit[i] = newfitness;
                    popnew[i] = tmp;
                    //updates the individual in pop (avoiding to recompute the objective function)
                    pop.set_xf(i,popnew[i],newfitness);

                    if ( newfitness[0] <= gbfit[0] ) {
                        /* if so...*/
                        gbfit=newfitness;                   /* reset gbfit to new low...*/
                        gbX=popnew[i];
                    }
                } else {
                    popnew[i] = popold[i];
                }
            } // End of one generation

            /* Save best population member of current iteration */
            gbIter = gbX;
            /* swap population arrays. New generation becomes old one */
            std::swap(popold, popnew);

            //9 - Check the exit conditions (every 40 generations)
            double dx = 0., df = 0.;
            if (gen % 40u == 0u) {
                best_idx = pop.best_idx();
                worst_idx = pop.worst_idx();
                for (decltype(dim) i = 0u; i < dim; ++i) {
                    dx += std::abs(pop.get_x()[worst_idx][i] - pop.get_x()[best_idx][i]);
                }
                if  (dx < m_xtol) {
                    if (m_verbosity > 0u) {
                        std::cout << "Exit condition -- xtol < " <<  m_xtol << std::endl;
                    }
                    return pop;
                }

                df = std::abs(pop.get_f()[worst_idx][0] - pop.get_f()[best_idx][0]);
                if (df < m_ftol) {
                    if (m_verbosity > 0u) {
                        std::cout << "Exit condition -- ftol < " <<  m_ftol << std::endl;
                    }
                    return pop;
                }
            }

            // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Every m_verbosity generations print a log line
                if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                    best_idx = pop.best_idx();
                    worst_idx = pop.worst_idx();
                    dx = 0.;
                    // The population flattness in chromosome
                    for (decltype(dim) i = 0u; i < dim; ++i) {
                        dx += std::abs(pop.get_x()[worst_idx][i] - pop.get_x()[best_idx][i]);
                    }
                    // The population flattness in fitness
                    df = std::abs(pop.get_f()[worst_idx][0] - pop.get_f()[best_idx][0]);
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7),"Gen:", std::setw(15), "Fevals:", std::setw(15), "Best:", std::setw(15), "dx:", std::setw(15), "df:",'\n');
                    }
                    print(std::setw(7),gen, std::setw(15), prob.get_fevals() - fevals0, std::setw(15), pop.get_f()[best_idx][0], std::setw(15), dx, std::setw(15), df,'\n');
                    ++count;
                    // Logs
                    m_log.push_back(log_line_type(gen, prob.get_fevals() - fevals0, pop.get_f()[best_idx][0], dx, df));
                }
            }

        } //end main DE iterations
        if (m_verbosity) {
            std::cout << "Exit condition -- generations > " <<  m_gen << std::endl;
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
        return "Differential Evolution";
    }
    /// Extra informations
    std::string get_extra_info() const
    {
        return "\tGenerations: " + std::to_string(m_gen) +
            "\n\tParameter F: " + std::to_string(m_F) +
            "\n\tParameter CR: " + std::to_string(m_CR) +
            "\n\tStrategy: " + std::to_string(m_strategy) +
            "\n\tStopping xtol: " + std::to_string(m_xtol) +
            "\n\tStopping ftol: " + std::to_string(m_ftol) +
            "\n\tVerbosity: " + std::to_string(m_verbosity) +
            "\n\tSeed: " + std::to_string(m_seed);
    }
    /// Get log
    const log_type& get_log() const {
        return m_log;
    }
    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_gen,m_F,m_CR,m_strategy,m_ftol,m_xtol,m_e,m_seed,m_verbosity,m_log);
    }
private:
    unsigned int                        m_gen;
    double                              m_F;
    double                              m_CR;
    unsigned int                        m_strategy;
    double                              m_ftol;
    double                              m_xtol;
    mutable detail::random_engine_type  m_e;
    unsigned int                        m_seed;
    unsigned int                        m_verbosity;
    mutable log_type                    m_log;
};

} //namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::de)

#endif
