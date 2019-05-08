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

#ifndef PYGMO_REGISTER_AP_HPP
#define PYGMO_REGISTER_AP_HPP

#include <boost/python/extract.hpp>
#include <boost/python/import.hpp>
#include <boost/python/scope.hpp>
#include <cstdint>
#include <string>
#include <unordered_set>

#include <pygmo/common_utils.hpp>
#include <pygmo/numpy.hpp>

namespace pygmo
{

namespace bp = boost::python;

inline void register_ap()
{
    // Import the numpy API.
    numpy_import_array();

    // Register the AP with pygmo by adding it to the AP list.
    auto &ap_set = *reinterpret_cast<std::unordered_set<std::string> *>(
        bp::extract<std::uintptr_t>(bp::import("pygmo").attr("core").attr("_ap_set_address"))());
    ap_set.insert(bp::extract<std::string>(bp::scope().attr("__name__"))());
}
} // namespace pygmo

#endif
