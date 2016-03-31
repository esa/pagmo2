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
            decision_vector     cur_x;
            /// Current decision vector (integer part)
            decision_vector_int     cur_xi;
            /// Current constraint vector.
            constraint_vector   cur_c;
            /// Current fitness vector.
            fitness_vector      cur_f;
        };

        /// Underlying container type.
        typedef std::vector<individual_type> container_type;

        /// Constructor (MA LO VOGLIAMO IL PROBLEMA QUI?? PERCHE NON RINUNCIARE E
        /// DEFINIRE INDIVIDUI SVINCOLATI DAL PROBLEMA?)
        explicit population(const pagmo::problem &, unsigned int = 0, const boost::uint32_t &seed = getSeed());
        
        /// Copy constructor
        population(const population &);

        /// TBD COME GESTIAMO IL SEED CONTROL?
        static boost::uint32_t getSeed(){
            return rng_generator::get<rng_uint32>()();
        }

        const individual_type &get_individual(const size_type &) const;
        const pagmo::problem &get_problem() const;
        void set_x(const size_type &, const decision_vector &);
        void set_xi(const size_type &, const decision_vector_int &);
        void push_back(const decision_vector &, const decision_vector_int & = decision_vector_int());
        void erase(const size_type &);
        size_type size() const;

    private:
        // Problem. (LO VOGLIAMO??)
        problem                         m_prob;
        // individuals.
        container_type                  m_container;
        // Double precision random number generator.
        mutable rng_double              m_drng;
        // uint32 random number generator.
        mutable rng_uint32              m_urng;
};

} // namespace pagmo


#endif
