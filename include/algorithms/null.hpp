#ifndef PAGMO_ALGORITHM_NULL_HPP
#define PAGMO_ALGORITHM_NULL_HPP

#include "../algorithm.hpp"

namespace pagmo
{

namespace algorithms
{

class null
{
    public:
        null():m_a(42.1) {}
        void evolve() const {};

        const double& get_a() const
        {
            return m_a;
        }

        template <typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_a);
        }
    private:
        double m_a;
};

}} //namespaces

PAGMO_REGISTER_ALGORITHM(pagmo::algorithms::null);

#endif
