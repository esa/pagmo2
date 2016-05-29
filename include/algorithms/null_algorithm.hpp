#ifndef PAGMO_ALGORITHMS_NULL_HPP
#define PAGMO_ALGORITHMS_NULL_HPP

#include "../detail/population_fwd.hpp"

namespace pagmo
{

class null_algorithm
{
    public:
        /// Constructor
        null_algorithm():m_a(42.1) {}

        /// Algorithm implementation
        population evolve(const population& pop) const {
            return pop;
        };

        /// Getter for the (irrelevant) algorithm parameter
        const double& get_a() const
        {
            return m_a;
        }

        /// Problem name
        std::string get_name() const
        {
            return "Null algorithm";
        }

        /// Extra informations
        std::string get_extra_info() const
        {
            return "\tUseless parameter: " + std::to_string(m_a);
        }

        template <typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_a);
        }
    private:
        double m_a;
};

} //namespaces

PAGMO_REGISTER_ALGORITHM(pagmo::null_algorithm)

#endif
