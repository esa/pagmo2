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

from __future__ import absolute_import as _ai

import unittest as _ut


class udi_01(object):

    def run_evolve(self, algo, pop):
        return algo.evolve(pop)

    def get_name(self):
        return "udi_01"

    def get_extra_info(self):
        return "extra bits"


class udi_02(object):
    pass


class island_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.core.island` class.

    """

    def runTest(self):
        self.run_basic_tests()
        self.run_concurrent_access_tests()

    def run_basic_tests(self):
        from .core import island, thread_island, null_algorithm, null_problem, de, rosenbrock
        isl = island()
        self.assertTrue(isl.get_algorithm().is_(null_algorithm))
        self.assertTrue(isl.get_population().problem.is_(null_problem))
        self.assertEqual(len(isl.get_population()), 0)
        isl = island(algo=de(), prob=rosenbrock(), size=10)
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(rosenbrock))
        self.assertEqual(len(isl.get_population()), 10)
        isl = island(prob=rosenbrock(), udi=thread_island(),
                     size=11, algo=de(), seed=15)
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(rosenbrock))
        self.assertEqual(len(isl.get_population()), 11)
        self.assertEqual(isl.get_population().get_seed(), 15)
        isl = island(prob=rosenbrock(), udi=udi_01(),
                     size=11, algo=de(), seed=15)
        self.assertEqual(isl.get_name(), "udi_01")
        self.assertEqual(isl.get_extra_info(), "extra bits")
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(rosenbrock))
        self.assertEqual(len(isl.get_population()), 11)
        self.assertRaises(NotImplementedError, lambda: island(prob=rosenbrock(), udi=udi_02(),
                                                              size=11, algo=de(), seed=15))

    def run_concurrent_access_tests(self):
        import threading as thr
