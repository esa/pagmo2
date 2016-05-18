#ifndef PAGMO_ALGORITHMS_SEA_HPP
#define PAGMO_ALGORITHMS_SEA_HPP

#include <iomanip>
#include <random>
#include <tuple>

#include "../algorithm.hpp"
#include "../io.hpp"
#include "../exceptions.hpp"
#include "../population.hpp"
#include "../rng.hpp"

namespace pagmo
{

class sea
{
    using log_line = std::tuple<unsigned int, double, double, unsigned int>;
    using log_type = std::vector<log_line>;

    public:
        /// Constructor
        sea(unsigned int gen = 1u, unsigned int seed = pagmo::random_device::next()):m_gen(gen),m_e(seed),m_seed(seed), m_log() {}

        /// Algorithm evolve method (juice implementation of the algorithm)
        population evolve(population pop) const {
            // We store some useful properties
            const auto &prob = pop.get_problem();
            const auto &dim = prob.get_nx();
            const auto &bounds = prob.get_bounds();
            const auto &lb = bounds.first;
            const auto &ub = bounds.second;

            // PREAMBLE-------------------------------------------------------------------------------------------------
            // We start by checking that the problem is suitable for this
            // particular algorithm.
            if (prob.get_nc() != 0u) {
                pagmo_throw(std::invalid_argument,"Non linear constraints detected. " + get_name() + " cannot deal with them");
            }
            if (prob.get_nf() != 1u) {
                pagmo_throw(std::invalid_argument,"Multiple objectives detected. " + get_name() + " cannot deal with them");
            }
            // Get out if there is nothing to do.
        	if (m_gen == 0u) {
        		return pop;
            }
            // ---------------------------------------------------------------------------------------------------------

            // Main loop
            // 1 - Compute the best and worst individual (index)
            auto best_idx = pop.get_best_idx();
            auto worst_idx = pop.get_worst_idx();
            unsigned int count = 1u; // regulates the screen output
            std::uniform_real_distribution<double> drng(0.,1.); // [0,1]

            for (unsigned int i = 1u; i <= m_gen; ++i) {
                vector_double offspring = pop.get_x()[best_idx];
                // 2 - Mutate the components (at least one) of the best
                auto mut = 0u;
                while(mut==0) {
                    for (vector_double::size_type j = 0u; j < dim; ++j) { // for each decision vector component
                        if (drng(m_e) < 1.0 / static_cast<double>(dim))
                        {
                            offspring[j] = std::uniform_real_distribution<double>(lb[j],ub[j])(m_e);
                            ++mut;
                        }
                    }
                }
                // 3 - Insert the offspring into the population if better
                auto offspring_f = prob.fitness(offspring);
                auto improvement = pop.get_f()[worst_idx][0] - offspring_f[0];
                if (improvement >= 0) {
                    pop.set_xf(worst_idx, offspring, offspring_f);
                    if (pop.get_f()[best_idx][0] - offspring_f[0] >= 0.) {
                        best_idx = worst_idx;
                    }
                    worst_idx = pop.get_worst_idx();
                    // Logs and prints (verbosity mode 1: a line is added everytime the population is improved by the offspring)
                    if (m_verbosity == 1u && improvement > 0.)
                    {
                        // Prints on screen
                        if (count % 50 == 1u) {
                            print("\n", std::setw(7),"Gen:", std::setw(15), "Best:", std::setw(15), "Improvement:", std::setw(15), "Mutations:",'\n');
                        }
                        print(std::setw(7),i, std::setw(15), pop.get_f()[best_idx][0], std::setw(15), improvement, std::setw(15), mut,'\n');
                        ++count;
                        // Logs
                        m_log.push_back(log_line(i, pop.get_f()[best_idx][0], improvement, mut));
                    }
                }
                // 4 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
                if (m_verbosity > 1u) {
                    // Every m_verbosity generations print a log line
                    if (i % m_verbosity == 1u) {
                        // Every 50 lines print the column names
                        if (count % 50 == 1u) {
                            print("\n", std::setw(7),"Gen:", std::setw(15), "Best:", std::setw(15), "Improvement:", std::setw(15), "Mutations:",'\n');
                        }
                        print(std::setw(7),i, std::setw(15), pop.get_f()[best_idx][0], std::setw(15), improvement, std::setw(15), mut,'\n');
                        ++count;
                        // Logs
                        m_log.push_back(log_line(i, pop.get_f()[best_idx][0], improvement, mut));
                    }
                }
            }
            return pop;
        };

        /// Set seed method
        void set_seed(unsigned int seed)
        {
            m_seed = seed;
        };
        // Sets the verbosity
        void set_verbosity(unsigned int level)
        {
            m_verbosity = level;
        };
        /// Problem name
        std::string get_name() const
        {
            return "(N+1)-EA Simple Evolutionary Algorithm";
        }

        /// Extra informations
        std::string get_extra_info() const
        {
            return "\tGenerations: " + std::to_string(m_gen) + "\n\tVerbosity: " + std::to_string(m_verbosity)
                + "\n\tSeed: " + std::to_string(m_seed);
        }
        /// Get log
        const log_type& get_log() const {
            return m_log;
        }

        template <typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_gen,m_e,m_seed,m_verbosity);
        }
    private:
        unsigned int                                     m_gen;
        mutable detail::random_engine_type               m_e;
        unsigned int                                     m_seed;
        unsigned int                                     m_verbosity = 0u;
        mutable log_type                                 m_log;
};

} //namespaces

PAGMO_REGISTER_ALGORITHM(pagmo::sea)

#endif
