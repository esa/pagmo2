#ifndef UDP_BASIC_HPP
#define UDP_BASIC_HPP

#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

struct udp_basic {
    pagmo::vector_double fitness(const pagmo::vector_double &dv) const
    {
        return {dv[0] * dv[0]};
    }
    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const
    {
        return {{-1}, {1}};
    }
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

PAGMO_S11N_PROBLEM_EXPORT(udp_basic)

#endif
