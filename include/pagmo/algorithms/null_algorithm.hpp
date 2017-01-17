/* Copyright 2017 PaGMO development team
This file is part of the PaGMO library.
The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:
  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.
or
  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.
or both in parallel, as here.
The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.
You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#ifndef PAGMO_ALGORITHMS_NULL_HPP
#define PAGMO_ALGORITHMS_NULL_HPP

#include "../algorithm.hpp"
#include "../detail/population_fwd.hpp"

namespace pagmo
{

class null_algorithm
{
public:
    /// Constructor
    null_algorithm() : m_a(42.1)
    {
    }

    /// Algorithm implementation
    population evolve(const population &pop) const
    {
        return pop;
    };

    /// Getter for the (irrelevant) algorithm parameter
    const double &get_a() const
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

} // namespaces

PAGMO_REGISTER_ALGORITHM(pagmo::null_algorithm)

#endif
