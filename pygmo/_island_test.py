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


class _udi_01(object):

    def run_evolve(self, algo, pop):
        return algo.evolve(pop)

    def get_name(self):
        return "udi_01"

    def get_extra_info(self):
        return "extra bits"


class _udi_02(object):
    pass


class _prob(object):

    def __init__(self, data):
        self.data = data

    def fitness(self, x):
        return [0.]

    def get_bounds(self):
        return ([0.], [1.])


class island_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.island` class.

    """

    def runTest(self):
        self.run_basic_tests()
        self.run_concurrent_access_tests()
        self.run_evolve_tests()
        self.run_get_busy_wait_tests()
        self.run_thread_safety_tests()
        self.run_io_tests()
        self.run_status_tests()

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
        isl = island(prob=rosenbrock(), udi=_udi_01(),
                     size=11, algo=de(), seed=15)
        self.assertEqual(isl.get_name(), "udi_01")
        self.assertEqual(isl.get_extra_info(), "extra bits")
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(rosenbrock))
        self.assertEqual(len(isl.get_population()), 11)
        self.assertRaises(NotImplementedError, lambda: island(prob=rosenbrock(), udi=_udi_02(),
                                                              size=11, algo=de(), seed=15))

    def run_concurrent_access_tests(self):
        import threading as thr
        from .core import island, de, rosenbrock
        isl = island(algo=de(), prob=rosenbrock(), size=10)

        def thread_func():
            for i in range(100):
                pop = isl.get_population()
                isl.set_population(pop)
                algo = isl.get_algorithm()
                isl.set_algorithm(algo)

        thr_list = [thr.Thread(target=thread_func) for i in range(4)]
        [_.start() for _ in thr_list]
        [_.join() for _ in thr_list]

    def run_evolve_tests(self):
        from .core import island, de, rosenbrock
        from copy import deepcopy
        isl = island(algo=de(), prob=rosenbrock(), size=25)
        isl.evolve(0)
        isl.wait_check()
        isl.evolve()
        isl.evolve()
        isl.wait_check()
        isl.evolve(20)
        isl.wait_check()
        for i in range(10):
            isl.evolve(20)
        isl2 = deepcopy(isl)
        isl2.wait_check()
        isl.wait_check()

    def run_status_tests(self):
        from . import island, de, rosenbrock, evolve_status
        isl = island(algo=de(), prob=rosenbrock(), size=3)
        isl.evolve(20)
        isl.wait()
        self.assertTrue(isl.status == evolve_status.idle_error)
        self.assertRaises(BaseException, lambda: isl.wait_check())
        self.assertTrue(isl.status == evolve_status.idle)

    def run_get_busy_wait_tests(self):
        from . import island, de, rosenbrock, evolve_status
        isl = island(algo=de(), prob=rosenbrock(), size=25)
        self.assertTrue(isl.status == evolve_status.idle)
        isl = island(algo=de(), prob=rosenbrock(), size=3)
        isl.evolve(20)
        self.assertRaises(BaseException, lambda: isl.wait_check())
        isl.evolve(20)
        isl.wait()

    def run_thread_safety_tests(self):
        from .core import island, de, rosenbrock
        from . import thread_safety as ts
        isl = island(algo=de(), prob=rosenbrock(), size=25)
        self.assertEqual(isl.get_thread_safety(), (ts.basic, ts.basic))

        class prob(object):

            def fitness(self, x):
                return [0]

            def get_bounds(self):
                return ([0.], [1.])

        isl = island(algo=de(), prob=prob(), size=25)
        self.assertEqual(isl.get_thread_safety(), (ts.basic, ts.none))

        class algo(object):

            def evolve(self, algo, pop):
                return pop

        isl = island(algo=algo(), prob=rosenbrock(), size=25)
        self.assertEqual(isl.get_thread_safety(), (ts.none, ts.basic))
        isl = island(algo=algo(), prob=prob(), size=25)
        self.assertEqual(isl.get_thread_safety(), (ts.none, ts.none))
        isl.evolve(20)
        self.assertRaises(BaseException, lambda: isl.wait_check())

    def run_io_tests(self):
        from .core import island, de, rosenbrock
        isl = island(algo=de(), prob=rosenbrock(), size=25)
        self.assertTrue(repr(isl) != "")
        self.assertTrue(isl.get_name() == "Thread island")
        self.assertTrue(isl.get_extra_info() == "")
        isl = island(algo=de(), prob=rosenbrock(), size=25, udi=_udi_01())
        self.assertTrue(repr(isl) != "")
        self.assertTrue(isl.get_name() == "udi_01")
        self.assertTrue(isl.get_extra_info() == "extra bits")

    def run_serialization_tests(self):
        from .core import island, de, rosenbrock
        from pickle import dumps, loads
        isl = island(algo=de(), prob=rosenbrock(), size=25)
        tmp = repr(isl)
        isl = loads(dumps(isl))
        self.assertEqual(tmp, repr(isl))


class mp_island_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.mp_island` class.

    """

    def __init__(self, level):
        _ut.TestCase.__init__(self)
        self._level = level

    def runTest(self):
        import sys
        import os
        # The mp island requires either Windows or at least Python 3.4.
        if os.name != 'nt' and (sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 4)):
            return

        self.run_basic_tests()

    def run_basic_tests(self):
        from .core import island, de, rosenbrock
        from . import mp_island
        from copy import copy, deepcopy
        from pickle import dumps, loads
        isl = island(algo=de(), prob=rosenbrock(), size=25, udi=mp_island())
        self.assertEqual(isl.get_name(), "Multiprocessing island")
        self.assertTrue(isl.get_extra_info() != "")
        self.assertTrue(mp_island.get_pool_size() > 0)
        mp_island.init_pool()
        self.assertRaises(TypeError, lambda: mp_island.init_pool("dasda"))
        self.assertRaises(ValueError, lambda: mp_island.init_pool(0))
        self.assertRaises(ValueError, lambda: mp_island.init_pool(-1))
        mp_island.resize_pool(6)
        isl.evolve(20)
        isl.evolve(20)
        mp_island.resize_pool(4)
        isl.wait_check()
        isl.evolve(20)
        isl.evolve(20)
        isl.wait()
        self.assertRaises(ValueError, lambda: mp_island.resize_pool(-1))
        self.assertRaises(TypeError, lambda: mp_island.resize_pool("dasda"))

        # Check the picklability of a problem storing a lambda.
        isl = island(algo=de(), prob=_prob(
            lambda x, y: x + y), size=25, udi=mp_island())
        isl.evolve()
        isl.wait_check()

        # Copy/deepcopy.
        isl2 = copy(isl)
        isl3 = deepcopy(isl)
        self.assertEqual(str(isl2), str(isl))
        self.assertEqual(str(isl3), str(isl))

        # Pickle.
        self.assertEqual(str(loads(dumps(isl))), str(isl))

        if self._level == 0:
            return

        # Check exception transport.
        for _ in range(1000):
            isl = island(algo=de(), prob=_prob(
                lambda x, y: x + y), size=2, udi=mp_island())
            isl.evolve()
            isl.wait()
            self.assertTrue("**error occurred**" in repr(isl))
            self.assertRaises(RuntimeError, lambda: isl.wait_check())


class ipyparallel_island_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.ipyparallel` class.

    """

    def __init__(self, level):
        _ut.TestCase.__init__(self)
        self._level = level

    def runTest(self):
        try:
            import ipyparallel
        except ImportError:
            return

        self.run_basic_tests()

    def run_basic_tests(self):
        from .core import island, de, rosenbrock
        from . import ipyparallel_island
        from copy import copy, deepcopy
        from pickle import dumps, loads
        to = .5
        try:
            isl = island(algo=de(), prob=rosenbrock(),
                         size=25, udi=ipyparallel_island(timeout=to))
        except OSError:
            return
        isl = island(algo=de(), prob=rosenbrock(),
                     size=25, udi=ipyparallel_island(timeout=to))
        isl = island(algo=de(), prob=rosenbrock(),
                     size=25, udi=ipyparallel_island(timeout=to + .3))
        self.assertEqual(isl.get_name(), "Ipyparallel island")
        self.assertTrue(isl.get_extra_info() != "")
        isl.evolve(20)
        isl.wait_check()
        isl.evolve(20)
        isl.evolve(20)
        isl.wait()

        # Check the picklability of a problem storing a lambda.
        isl = island(algo=de(), prob=_prob(lambda x, y: x + y),
                     size=25, udi=ipyparallel_island(timeout=to + .3))
        isl.evolve()
        isl.wait_check()

        # Copy/deepcopy.
        isl2 = copy(isl)
        isl3 = deepcopy(isl)
        self.assertEqual(str(isl2.get_population()), str(isl.get_population()))
        self.assertEqual(str(isl2.get_algorithm()), str(isl.get_algorithm()))
        self.assertEqual(str(isl2.get_name()), str(isl.get_name()))
        self.assertEqual(str(isl3.get_population()), str(isl.get_population()))
        self.assertEqual(str(isl3.get_algorithm()), str(isl.get_algorithm()))
        self.assertEqual(str(isl3.get_name()), str(isl.get_name()))

        # Pickle.
        pisl = loads(dumps(isl))
        self.assertEqual(str(pisl.get_population()), str(isl.get_population()))
        self.assertEqual(str(pisl.get_algorithm()), str(isl.get_algorithm()))
        self.assertEqual(str(pisl.get_name()), str(isl.get_name()))

        if self._level == 0:
            return

        # Check exception transport.
        for _ in range(1000):
            isl = island(algo=de(), prob=_prob(
                lambda x, y: x + y), size=2, udi=ipyparallel_island(timeout=to + .3))
            isl.evolve()
            isl.wait()
            self.assertTrue("**error occurred**" in repr(isl))
            self.assertRaises(RuntimeError, lambda: isl.wait_check())
