#ifndef PAGMO_POPULATION_H
#define PAGMO_POPULATION_H

#include <vector>

#include "rng.h"

namespace pagmo
{

typedef std::vector<double> decision_vector
typedef std::vector<long long> decision_vector_int
typedef std::vector<double> constraint_vector
typedef std::vector<double> fitness_vector

class population
{
    public:
        /// Individual
        struct individual_type
        {
            /// Current decision vector (continuous part)
            decision_vector         cur_x;
            /// Current decision vector (integer part)
            decision_vector_int     cur_xi;
            /// Current constraint vector.
            constraint_vector       cur_c;
            /// Current fitness vector.
            fitness_vector          cur_f;
        };

        /// Underlying container type.
        typedef std::vector<individual_type> container_type;

        /// Constructors
        explicit population(const pagmo::problem &p, unsigned int size = 0, unsigned int seed = pagmo::random_device::next() : m_prob(p), m_e(seed), m_seed(seed)
        {
            // Store problem sizes temporarily.
            const fitness_vector::size_type f_size = prob->get_f_dimension();
            const constraint_vector::size_type c_size = m_prob->get_c_dimension();
            const decision_vector::size_type p_size = m_prob->get_dimension();
            for (size_type i = 0; i < size; ++i) {
                // Push back an empty individual.
                m_container.push_back(individual_type());
                // Resize individual's elements.
                m_container.back().x.resize(p_size);
                m_container.back().x_i.resize(p_size);
                m_container.back().c.resize(c_size);
                m_container.back().f.resize(f_size);
                // Initialise randomly the individual.
                reinit(i);
            }
        }
        
        /// Copy constructor
        population(const population &);


        const individual_type &get_individual(const size_type &) const;
        const pagmo::problem &get_problem() const;
        void set_ind(const size_type &, const decision_vector &, const decision_vector_int & = decision_vector_int());
        void set_ind(const size_type &, const decision_vector_int &, const decision_vector & = decision_vector());

        void push_back(const decision_vector &, const decision_vector_int & = decision_vector_int());
        void push_back(const decision_vector_int &, const decision_vector & = decision_vector());
        void erase(const size_type &);
        
        size_type size() const;

    private:
        // Problem. (LO VOGLIAMO??)
        problem                         m_prob;
        // individuals.
        container_type                  m_container;
        // Random engine
        random_engine_type              m_e;
        // Seed 
        unsigned int                    m_seed;
};

} // namespace pagmo


#endif
