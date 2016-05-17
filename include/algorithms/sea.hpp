#ifndef PAGMO_ALGORITHMS_SEA_HPP
#define PAGMO_ALGORITHMS_SEA_HPP

#include "../algorithm.hpp"
#include "../detail/population_fwd.hpp"

namespace pagmo
{

class sea
{
    public:
        /// Constructor
        sea(unsigned int gen = 1u, unsigned int seed = pagmo::random_device::next()):m_gen(gen),m_e(seed),m_seed(seed) {}

        /// Algorithm implementation
        population evolve(population pop) const {
            // We store some useful properties
            const auto &prob = pop.get_problem();
            const auto &dim = pop.get_nx();
            const auto &bounds = prob.get_bounds();
            const auto &lb = bounds.first();
            const auto &ub = bounds.second();

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
        		return;
            }
            // ---------------------------------------------------------------------------------------------------------

            std::uniform_real_distribution<double> drng(0.,1.);
            double new_gene(0.);

            // Juice implementation
            for (unsigned int i = 0u; i < m_gen; ++i) {
                // 1 - Extract the best decision vector so far
                auto best_idx = pop.get_best_idx();
                vector_double offspring = pop.get_x()[best_idx];
                // 2 - Mutate its components
                for (decltype(dim) j = 0u; j < dim;++j) { // for each decision vector component
                    if (drng(m_e) < 1.0 / dim)
                    {
                        new_gene = std::uniform_real_distribution<double>(lb[j],ub[j])(m_e);
                        offspring[j] = new_gene;
                    }
                }
                // 3 - Insert the offspring into the population if better
                auto worst_idx = pop.get_worst_idx();
                auto offspring_f = prob.fitness(offspring, );
                if (offspring_f <= pop.get_f()[worst_idx]) {
                    pop.set_xf(worst_idx, offspring, offspring_f);
                }
            }
            return pop;
        };

        /// Set seed method
        void set_seed(unsigned int seed)
        {
            m_seed = seed;
        };
        /// Problem name
        std::string get_name() const
        {
            return "(N+1)-EA Simple Evolutionary Algorithm";
        }

        /// Extra informations
        std::string get_extra_info() const
        {
            return "\tGenerations: " + std::to_string(m_gen);
        }

        template <typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_gen,m_e,m_seed);
        }
    private:
        unsigned int                                     m_gen;
        mutable detail::random_engine_type               m_e;
        unsigned int                                     m_seed;
};

} //namespaces

PAGMO_REGISTER_ALGORITHM(pagmo::null_algorithm)

#endif
