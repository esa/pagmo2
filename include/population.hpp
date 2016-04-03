#ifndef PAGMO_POPULATION_H
#define PAGMO_POPULATION_H

#include <vector>

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
        explicit population(const pagmo::problem &p, unsigned int size = 0, unsigned int seed = std::random_device{}()) : m_prob(p), m_e(seed), m_seed(seed);
        
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
        problem                         m_  prob;
        // individuals.
        container_type                  m_container;
        // Random engine
        random_engine_type              m_e;
        // Seed 
        unsigned int                    m_seed;
};

} // namespace pagmo


#endif
