#ifndef UDA_BASIC_HPP
#define UDA_BASIC_HPP

#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>

struct uda_basic {
    pagmo::population evolve(const pagmo::population &pop) const
    {
        return pop;
    }
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};

PAGMO_REGISTER_ALGORITHM(uda_basic)

#endif
