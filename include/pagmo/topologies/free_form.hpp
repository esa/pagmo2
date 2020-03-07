/* Copyright 2017-2020 PaGMO development team

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

#ifndef PAGMO_TOPOLOGIES_FREE_FORM_HPP
#define PAGMO_TOPOLOGIES_FREE_FORM_HPP

#include <string>
#include <type_traits>

#include <pagmo/detail/free_form_fwd.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/topologies/base_bgl_topology.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/type_traits.hpp>

namespace pagmo
{

// Free-form topology.
class PAGMO_DLL_PUBLIC free_form : public base_bgl_topology
{
public:
    free_form();
    free_form(const free_form &);
    free_form(free_form &&) noexcept;

    explicit free_form(bgl_graph_t);
    explicit free_form(const topology &);
    template <typename T,
              enable_if_t<detail::conjunction<detail::negation<std::is_same<T, free_form>>, is_udt<T>>::value, int> = 0>
    explicit free_form(const T &t) : free_form(topology(t))
    {
    }

    void push_back();

    std::string get_name() const;

    template <typename Archive>
    void serialize(Archive &, unsigned);
};

} // namespace pagmo

PAGMO_S11N_TOPOLOGY_EXPORT_KEY(pagmo::free_form)

#endif
