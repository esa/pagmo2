#ifndef PAGMO_ALGORITHM_DE_HPP
#define PAGMO_ALGORITHM_DE_HPP

#include "../algorithm.hpp"

namespace pagmo
{

namespace algorithms
{

class de
{
    public:
        de():m_a(42.1) {}
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

}

}

PAGMO_REGISTER_ALGORITHM(pagmo::algorithms::de);

#endif
