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

#ifndef PYGMO_SR_POLICY_ADD_RATE_CONSTRUCTOR_HPP
#define PYGMO_SR_POLICY_ADD_RATE_CONSTRUCTOR_HPP

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/object.hpp>

#include <pygmo/common_utils.hpp>

namespace pygmo
{

namespace detail
{

namespace bp = boost::python;

// An helper to add a constructor from a migration rate to a
// replacement/selection policy.
template <typename Pol>
inline void sr_policy_add_rate_constructor(bp::class_<Pol> &c)
{
    c.def("__init__",
          bp::make_constructor(
              lcast([](const bp::object &o) -> Pol * {
                  if (pygmo::isinstance(o, pygmo::builtin().attr("int"))) {
                      const int r = bp::extract<int>(o);
                      return ::new Pol(r);
                  } else if (pygmo::isinstance(o, pygmo::builtin().attr("float"))) {
                      const double r = bp::extract<double>(o);
                      return ::new Pol(r);
                  } else {
                      pygmo_throw(PyExc_TypeError,
                                  ("cannot construct a replacement/selection policy from a migration rate of type '"
                                   + str(type(o)) + "': the migration rate must be an integral or floating-point value")
                                      .c_str());
                  }
              }),
              bp::default_call_policies(), (bp::arg("rate"))));
}

} // namespace detail

} // namespace pygmo

#endif
