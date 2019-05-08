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
    void serialize(Archive &, unsigned)
    {
    }
};

PAGMO_S11N_ALGORITHM_EXPORT(uda_basic)

#endif
