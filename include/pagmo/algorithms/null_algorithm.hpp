/* Copyright 2017-2018 PaGMO development team

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

#ifndef PAGMO_ALGORITHMS_NULL_ALGORITHM_HPP
#define PAGMO_ALGORITHMS_NULL_ALGORITHM_HPP

#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>

namespace pagmo
{

/// Null algorithm
/**
 * This algorithm is used to implement the default constructors of pagmo::algorithm and of the meta-algorithms.
 */
struct PAGMO_DLL_PUBLIC null_algorithm {
    // Evolve method.
    population evolve(const population &) const;
    /// Algorithm name.
    /**
     * @return <tt>"Null algorithm"</tt>.
     */
    std::string get_name() const
    {
        return "Null algorithm";
    }
    // Serialization support.
    template <typename Archive>
    void serialize(Archive &, unsigned);
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::null_algorithm)

#endif
