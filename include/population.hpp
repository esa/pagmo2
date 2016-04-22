#ifndef PAGMO_POPULATION_H
#define PAGMO_POPULATION_H

#include <iostream>
#include <random>
#include <vector>

#include "problems/null_problem.hpp"
#include "rng.hpp"
#include "serialization.hpp"
#include "types.hpp"

namespace pagmo
{

class population
{
    private:
        /// Individual
        struct individual
        {
            individual(const vector_double &fit, const vector_double &dv, const detail::random_engine_type::result_type& ind_id)
            : f(fit), x(dv), ID(ind_id) {}
            // fitness
            vector_double f;
            // decision vector
            vector_double x;
            // identity
            unsigned long long ID;
            // Streaming operator for the struct pagmo::problem::individual
            friend std::ostream &operator<<(std::ostream &os, const individual &p)
            {
                stream(os, "\tID:\t\t\t", p.ID, '\n');
                stream(os, "\tDecision vector:\t", p.x, '\n');
                stream(os, "\tFitness vector:\t\t", p.f, '\n');
                return os;
            }
        };
    public:
        /// Default constructor
        population() : m_prob(null_problem{}), m_container(), m_e(0u), m_seed(0u) {}

        /// Constructor
        explicit population(const pagmo::problem &p, std::vector<individual>::size_type size = 0u, unsigned int seed = pagmo::random_device::next()) : m_prob(p), m_e(seed), m_seed(seed)
        {
            for (decltype(size) i = 0u; i < size; ++i) {
                push_back(random_decision_vector());
            }
        }

        // Creates an individual from a decision vector and appends it
        // to the population
        void push_back(const vector_double &x)
        {
            // Do we call problem::check_decision_vector here?
            auto new_id = std::uniform_int_distribution<unsigned long long>()(m_e);
            m_container.push_back(individual{x,m_prob.fitness(x),new_id});
        }

        // Creates a random decision_vector within the problem bounds
        vector_double random_decision_vector() const
        {
            auto dim = m_prob.get_nx();
            auto bounds = m_prob.get_bounds();
            vector_double retval(dim);
            for (decltype(dim) i = 0u; i < dim; ++i) {
                std::uniform_real_distribution<double> dis(bounds.first[i], bounds.second[i]);
                retval[i] = dis(m_e);
            }
            return retval;
        }

        // Sets the seed of the population random engine
        void set_seed(unsigned int seed)
        {
            m_seed = seed;
            m_e.seed(seed);
        }

        // Gets the the seed of the population random engine
        unsigned int get_seed() const
        {
            return m_seed;
        }

        // Number of individuals in the population
        std::vector<individual>::size_type size() const
        {
            return m_container.size();
        }

        // Serialization.
        template <typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_prob, m_container, m_e, m_seed);
        }

        // Streaming operator for the class pagmo::problem
        friend std::ostream &operator<<(std::ostream &os, const population &p)
        {
            stream(os, p.m_prob, '\n');
            stream(os, "Population size: ",p.size(),"\n\n");
            stream(os, "List of individuals: ",'\n');
            for (decltype(p.m_container.size()) i=0u; i<p.m_container.size(); ++i) {
                stream(os, "#", i, ":\n");
                stream(os, p.m_container[i], '\n');
            }
            return os;
        }

    private:
        // Problem.
        problem                             m_prob;
        // Individuals.
        std::vector<individual>             m_container;
        // Random engine.
        mutable detail::random_engine_type  m_e;
        // Seed.
        unsigned int                        m_seed;
};

} // namespace pagmo


#endif
