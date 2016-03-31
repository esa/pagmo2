#ifndef PAGMO_POPULATION_H
#define PAGMO_POPULATION_H

#include <vector>

namespace pagmo
{

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
    /// Individual id.

};

typedef std::vector<double> decision_vector
typedef std::vector<long long> decision_vector_int
typedef std::vector<double> constraint_vector
typedef std::vector<double> fitness_vector

class population
{
    public:
        /// Underlying container type.
        typedef std::vector<individual_type> container_type;
        typedef container_type::size_type size_type;

        explicit population(container_type = container_type());
        
        /// Copy constructor
        population(const population &);

        const individual_type &get_individual(const size_type &) const;
        void set_individual(const size_type &, const individual_type &);
        void push_back(const individual_type &);
        size_type size() const;

    private:
        // individuals.
        container_type                  m_container;
};

} // namespace pagmo


#endif
