# -*- coding: utf-8 -*-

# Copyright 2017 PaGMO development team
#
# This file is part of the PaGMO library.
#
# The PaGMO library is free software; you can redistribute it and/or modify
# it under the terms of either:
#
#   * the GNU Lesser General Public License as published by the Free
#     Software Foundation; either version 3 of the License, or (at your
#     option) any later version.
#
# or
#
#   * the GNU General Public License as published by the Free Software
#     Foundation; either version 3 of the License, or (at your option) any
#     later version.
#
# or both in parallel, as here.
#
# The PaGMO library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received copies of the GNU General Public License and the
# GNU Lesser General Public License along with the PaGMO library.  If not,
# see https://www.gnu.org/licenses/.

# for python 2.0 compatibility
from __future__ import absolute_import as _ai

from ipyparallel import Client as _Client


class ipyparallel_island(object):

    def __init__(self):
        self._rc = _Client()
        self._rc[:].use_dill().get()
        # Setup of the ipyparallel bits.
        self._lview = self._rc.load_balanced_view()

    def __copy__(self):
        return ipyparallel_island()

    def __deepcopy__(self, d):
        return ipyparallel_island()

    def run_evolve(self, algo, pop):
        def evolve_func(algo, pop):
            return algo.evolve(pop)

        return self._lview.apply_sync(evolve_func, algo, pop)

    def get_name(self):
        return "Ipyparallel island"
