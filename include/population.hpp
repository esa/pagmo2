#ifndef PAGMO_POPULATION_H
#define PAGMO_POPULATION_H

#include <random>
#include <vector>

#include "rng.hpp"
#include "problems/null_problem.hpp"
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

            // Human readable representation of an individual
            std::string human_readable() const
            {
                std::ostringstream oss;
                stream(oss, "\tID:\t\t\t", ID, '\n');
                stream(oss, "\tDecision vector:\t", x, '\n');
                stream(oss, "\tFitness vector:\t\t", f, '\n');
                return oss.str();
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

        // Human readable representation of the population
        std::string human_readable() const
        {
            std::ostringstream oss;
            print(m_prob, '\n');
            print("Population size: ",size(),"\n\n");
            print("List of individuals: ",'\n');
            for (auto i=0u; i<m_container.size(); ++i) {
                print("#", i, ":\n");
                print(m_container[i].human_readable(), '\n');
            }
            return oss.str();
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

// Streaming operator for the class pagmo::problem
std::ostream &operator<<(std::ostream &os, const population &p)
{
    os << p.human_readable() << '\n';
    return os;
}

} // namespace pagmo


#endif
