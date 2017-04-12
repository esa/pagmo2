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


class _algo(object):

    def evolve(self, pop):
        return pop


class algorithm_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.algorithm` class.

    """

    def runTest(self):
        self.run_basic_tests()
        self.run_extract_tests()
        self.run_seed_tests()
        self.run_verbosity_tests()
        self.run_name_info_tests()
        self.run_thread_safety_tests()
        self.run_pickle_tests()

    def run_basic_tests(self):
        # Tests for minimal algorithm, and mandatory methods.
        from numpy import all, array
        from .core import algorithm, de, population, null_problem, null_algorithm
        from . import thread_safety as ts
        # Def construction.
        a = algorithm()
        self.assertTrue(a.extract(null_algorithm) is not None)
        self.assertTrue(a.extract(de) is None)

        # First a few non-algos.
        self.assertRaises(NotImplementedError, lambda: algorithm(1))
        self.assertRaises(NotImplementedError,
                          lambda: algorithm("hello world"))
        self.assertRaises(NotImplementedError, lambda: algorithm([]))
        self.assertRaises(TypeError, lambda: algorithm(int))
        # Some algorithms missing methods, wrong arity, etc.

        class na0(object):
            pass
        self.assertRaises(NotImplementedError, lambda: algorithm(na0()))

        class na1(object):

            evolve = 45
        self.assertRaises(NotImplementedError, lambda: algorithm(na1()))

        # The minimal good citizen.
        glob = []

        class a(object):

            def __init__(self, g):
                self.g = g

            def evolve(self, pop):
                self.g.append(1)
                return pop
        a_inst = a(glob)
        algo = algorithm(a_inst)

        # Test the keyword arg.
        algo = algorithm(uda=de())
        algo = algorithm(uda=a_inst)

        # Check a few algo properties.
        self.assertEqual(algo.is_stochastic(), False)
        self.assertEqual(algo.has_set_seed(), False)
        self.assertEqual(algo.has_set_verbosity(), False)
        self.assertEqual(algo.get_thread_safety(), ts.none)
        self.assertEqual(algo.get_extra_info(), "")
        self.assertRaises(NotImplementedError, lambda: algo.set_seed(123))
        self.assertRaises(NotImplementedError, lambda: algo.set_verbosity(1))
        self.assertTrue(algo.extract(int) is None)
        self.assertTrue(algo.extract(de) is None)
        self.assertFalse(algo.extract(a) is None)
        self.assertTrue(algo.is_(a))
        self.assertTrue(isinstance(algo.evolve(population()), population))
        # Assert that the global variable was copied into p, not simply
        # referenced.
        self.assertEqual(len(glob), 0)
        self.assertEqual(len(algo.extract(a).g), 1)
        algo = algorithm(de())
        self.assertEqual(algo.is_stochastic(), True)
        self.assertEqual(algo.has_set_seed(), True)
        self.assertEqual(algo.has_set_verbosity(), True)
        self.assertEqual(algo.get_thread_safety(), ts.basic)
        self.assertTrue(algo.get_extra_info() != "")
        self.assertTrue(algo.extract(int) is None)
        self.assertTrue(algo.extract(a) is None)
        self.assertFalse(algo.extract(de) is None)
        self.assertTrue(algo.is_(de))
        algo.set_seed(123)
        algo.set_verbosity(0)
        self.assertTrue(isinstance(algo.evolve(
            population(null_problem(), 5)), population))
        # Wrong retval for evolve().

        class a(object):

            def evolve(self, pop):
                return 3
        algo = algorithm(a())
        self.assertRaises(TypeError, lambda: algo.evolve(
            population(null_problem(), 5)))

    def run_extract_tests(self):
        from .core import algorithm, _test_algorithm, mbh
        import sys

        # First we try with a C++ test algo.
        p = algorithm(_test_algorithm())
        # Verify the refcount of p is increased after extract().
        rc = sys.getrefcount(p)
        tprob = p.extract(_test_algorithm)
        self.assert_(sys.getrefcount(p) == rc + 1)
        del tprob
        self.assert_(sys.getrefcount(p) == rc)
        # Verify we are modifying the inner object.
        p.extract(_test_algorithm).set_n(5)
        self.assert_(p.extract(_test_algorithm).get_n() == 5)
        # Chain extracts.
        t = mbh(_test_algorithm(), stop=5, perturb=[.4])
        pt = algorithm(t)
        rc = sys.getrefcount(pt)
        talgo = pt.extract(mbh)
        # Verify that extraction of mbh from the algo
        # increases the refecount of pt.
        self.assert_(sys.getrefcount(pt) == rc + 1)
        # Extract the _test_algorithm from mbh.
        rc2 = sys.getrefcount(talgo)
        ttalgo = talgo.inner_algorithm.extract(_test_algorithm)
        # The refcount of pt is not affected.
        self.assert_(sys.getrefcount(pt) == rc + 1)
        # The refcount of talgo has increased.
        self.assert_(sys.getrefcount(talgo) == rc2 + 1)
        del talgo
        # We can still access ttalgo.
        self.assert_(ttalgo.get_n() == 1)
        self.assert_(sys.getrefcount(pt) == rc + 1)
        del ttalgo
        # Now the refcount of pt decreases, because deleting
        # ttalgo eliminates the last ref to talgo, which in turn
        # decreases the refcount of pt.
        self.assert_(sys.getrefcount(pt) == rc)

        class talgorithm(object):

            def __init__(self):
                self._n = 1

            def get_n(self):
                return self._n

            def set_n(self, n):
                self._n = n

            def evolve(self, pop):
                return pop

        # Test with Python algo.
        p = algorithm(talgorithm())
        rc = sys.getrefcount(p)
        talgo = p.extract(talgorithm)
        # Reference count does not increase because
        # talgorithm is stored as a proper Python object
        # with its own refcount.
        self.assertTrue(sys.getrefcount(p) == rc)
        self.assertTrue(talgo.get_n() == 1)
        talgo.set_n(12)
        self.assert_(p.extract(talgorithm).get_n() == 12)

    def run_seed_tests(self):
        from .core import algorithm

        class a(object):

            def evolve(self, pop):
                return pop

        self.assertTrue(not algorithm(a()).has_set_seed())
        self.assertRaises(NotImplementedError,
                          lambda: algorithm(a()).set_seed(12))

        class a(object):

            def evolve(self, pop):
                return pop

            def has_set_seed(self):
                return True

        self.assertTrue(not algorithm(a()).has_set_seed())
        self.assertRaises(NotImplementedError,
                          lambda: algorithm(a()).set_seed(12))

        class a(object):

            def evolve(self, pop):
                return pop

            def set_seed(self, seed):
                pass

        self.assertTrue(algorithm(a()).has_set_seed())
        algorithm(a()).set_seed(87)

        class a(object):

            def evolve(self, pop):
                return pop

            def set_seed(self, seed):
                pass

            def has_set_seed(self):
                return False

        self.assertTrue(not algorithm(a()).has_set_seed())

        class a(object):

            def evolve(self, pop):
                return pop

            def set_seed(self, seed):
                pass

            def has_set_seed(self):
                return True

        self.assert_(algorithm(a()).has_set_seed())
        algorithm(a()).set_seed(0)
        algorithm(a()).set_seed(87)
        self.assertRaises(OverflowError, lambda: algorithm(a()).set_seed(-1))

    def run_verbosity_tests(self):
        from .core import algorithm

        class a(object):

            def evolve(self, pop):
                return pop

        self.assertTrue(not algorithm(a()).has_set_verbosity())
        self.assertRaises(NotImplementedError,
                          lambda: algorithm(a()).set_verbosity(12))

        class a(object):

            def evolve(self, pop):
                return pop

            def has_set_verbosity(self):
                return True

        self.assertTrue(not algorithm(a()).has_set_verbosity())
        self.assertRaises(NotImplementedError,
                          lambda: algorithm(a()).set_verbosity(12))

        class a(object):

            def evolve(self, pop):
                return pop

            def set_verbosity(self, level):
                pass

        self.assertTrue(algorithm(a()).has_set_verbosity())
        algorithm(a()).set_verbosity(87)

        class a(object):

            def evolve(self, pop):
                return pop

            def set_verbosity(self, level):
                pass

            def has_set_verbosity(self):
                return False

        self.assertTrue(not algorithm(a()).has_set_verbosity())

        class a(object):

            def evolve(self, pop):
                return pop

            def set_verbosity(self, level):
                pass

            def has_set_verbosity(self):
                return True

        self.assert_(algorithm(a()).has_set_verbosity())
        algorithm(a()).set_verbosity(0)
        algorithm(a()).set_verbosity(87)
        self.assertRaises(
            OverflowError, lambda: algorithm(a()).set_verbosity(-1))

    def run_name_info_tests(self):
        from .core import algorithm

        class a(object):

            def evolve(self, pop):
                return pop

        algo = algorithm(a())
        self.assert_(algo.get_name() != '')
        self.assert_(algo.get_extra_info() == '')

        class a(object):

            def evolve(self, pop):
                return pop

            def get_name(self):
                return 'pippo'

        algo = algorithm(a())
        self.assert_(algo.get_name() == 'pippo')
        self.assert_(algo.get_extra_info() == '')

        class a(object):

            def evolve(self, pop):
                return pop

            def get_extra_info(self):
                return 'pluto'

        algo = algorithm(a())
        self.assert_(algo.get_name() != '')
        self.assert_(algo.get_extra_info() == 'pluto')

        class a(object):

            def evolve(self, pop):
                return pop

            def get_name(self):
                return 'pippo'

            def get_extra_info(self):
                return 'pluto'

        algo = algorithm(a())
        self.assert_(algo.get_name() == 'pippo')
        self.assert_(algo.get_extra_info() == 'pluto')

    def run_thread_safety_tests(self):
        from .core import algorithm, de, _tu_test_algorithm, mbh
        from . import thread_safety as ts

        class a(object):

            def evolve(self, pop):
                return pop

        self.assertTrue(algorithm(a()).get_thread_safety() == ts.none)
        self.assertTrue(algorithm(de()).get_thread_safety() == ts.basic)
        self.assertTrue(
            algorithm(_tu_test_algorithm()).get_thread_safety() == ts.none)
        self.assertTrue(
            algorithm(mbh(_tu_test_algorithm(), stop=5, perturb=.4)).get_thread_safety() == ts.none)
        self.assertTrue(
            algorithm(mbh(a(), stop=5, perturb=.4)).get_thread_safety() == ts.none)
        self.assertTrue(
            algorithm(mbh(de(), stop=5, perturb=.4)).get_thread_safety() == ts.basic)

    def run_pickle_tests(self):
        from .core import algorithm, de, mbh
        from pickle import dumps, loads
        a_ = algorithm(de())
        a = loads(dumps(a_))
        self.assertEqual(repr(a), repr(a_))
        self.assertTrue(a.is_(de))
        a_ = algorithm(mbh(de(), 10, .1))
        a = loads(dumps(a_))
        self.assertEqual(repr(a), repr(a_))
        self.assertTrue(a.is_(mbh))
        self.assertTrue(a.extract(mbh).inner_algorithm.is_(de))

        a_ = algorithm(_algo())
        a = loads(dumps(a_))
        self.assertEqual(repr(a), repr(a_))
        self.assertTrue(a.is_(_algo))
        a_ = algorithm(mbh(_algo(), 10, .1))
        a = loads(dumps(a_))
        self.assertEqual(repr(a), repr(a_))
        self.assertTrue(a.is_(mbh))
        self.assertTrue(a.extract(mbh).inner_algorithm.is_(_algo))
