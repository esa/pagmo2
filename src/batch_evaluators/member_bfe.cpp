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

#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/bfe_impl.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

// Call operator.
vector_double member_bfe::operator()(const problem &p, const vector_double &dvs) const
{
    return detail::prob_invoke_mem_batch_fitness(p, dvs);
}

// Serialization support.
template <typename Archive>
void member_bfe::serialize(Archive &, unsigned)
{
}

} // namespace pagmo

PAGMO_S11N_BFE_IMPLEMENT(pagmo::member_bfe)
