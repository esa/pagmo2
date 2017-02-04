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


class core_test_case(_ut.TestCase):
    """Test case for core PyGMO functionality.

    """

    def runTest(self):
        import sys
        from numpy import random, all, array
        from .core import _builtin, _test_to_vd, _type, _str, _callable, _deepcopy, _test_object_serialization as tos
        if sys.version_info[0] < 3:
            import __builtin__ as b
        else:
            import builtins as b
        self.assertEqual(b, _builtin())
        self.assert_(_test_to_vd([], 0))
        self.assert_(_test_to_vd((), 0))
        self.assert_(_test_to_vd(array([]), 0))
        self.assert_(_test_to_vd([0], 1))
        self.assert_(_test_to_vd((0,), 1))
        self.assert_(_test_to_vd(array([0]), 1))
        self.assert_(_test_to_vd([0.], 1))
        self.assert_(_test_to_vd((0.,), 1))
        self.assert_(_test_to_vd(array([0.]), 1))
        self.assert_(_test_to_vd([0, 1.], 2))
        self.assert_(_test_to_vd([0, 1], 2))
        self.assert_(_test_to_vd((0., 1.), 2))
        self.assert_(_test_to_vd((0., 1), 2))
        self.assert_(_test_to_vd(array([0., 1.]), 2))
        self.assert_(_test_to_vd(array([0, 1]), 2))
        self.assertEqual(type(int), _type(int))
        self.assertEqual(str(123), _str(123))
        self.assertEqual(callable(1), _callable(1))
        self.assertEqual(callable(lambda _: None), _callable(lambda _: None))
        l = [1, 2, 3, ["abc"]]
        self.assert_(id(l) != id(_deepcopy(l)))
        self.assert_(id(l[3]) != id(_deepcopy(l)[3]))
        self.assertEqual(tos(l), l)
        self.assertEqual(tos({'a': l, 3: "Hello world"}),
                         {'a': l, 3: "Hello world"})
        a = random.rand(3, 2)
        self.assert_(all(tos(a) == a))


class problem_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.core.problem` class.

    """

    def runTest(self):
        self.run_basic_tests()
        self.run_extract_tests()
        self.run_nobj_tests()
        self.run_nec_nic_tests()
        self.run_has_gradient_tests()
        self.run_gradient_tests()

    def run_basic_tests(self):
        # Tests for minimal problem, and mandatory methods.
        from numpy import all, array
        from .core import problem
        # First a few non-problems.
        self.assertRaises(TypeError, lambda: problem(1))
        self.assertRaises(TypeError, lambda: problem("hello world"))
        self.assertRaises(TypeError, lambda: problem([]))
        self.assertRaises(TypeError, lambda: problem(int))
        # Some problems missing methods, wrong arity, etc.

        class np0(object):

            def fitness(self, a):
                return [1]
        self.assertRaises(TypeError, lambda: problem(np0))

        class np1(object):

            def get_bounds(self):
                return ([0], [1])
        self.assertRaises(TypeError, lambda: problem(np1))

        class np2(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a, b):
                return [42]
        self.assertRaises(TypeError, lambda: problem(np2))

        class np3(object):

            def get_bounds(self, a):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]
        self.assertRaises(TypeError, lambda: problem(np3))
        # The minimal good citizen.
        glob = []

        class p(object):

            def __init__(self, g):
                self.g = g

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                self.g.append(1)
                return [42]
        p_inst = p(glob)
        prob = problem(p_inst)
        # Check a few problem properties.
        self.assertEqual(prob.get_nobj(), 1)
        self.assert_(isinstance(prob.get_bounds(), tuple))
        self.assert_(all(prob.get_bounds()[0] == [0, 0]))
        self.assert_(all(prob.get_bounds()[1] == [1, 1]))
        self.assertEqual(prob.get_nx(), 2)
        self.assertEqual(prob.get_nf(), 1)
        self.assertEqual(prob.get_nec(), 0)
        self.assertEqual(prob.get_nic(), 0)
        self.assert_(not prob.has_gradient())
        self.assert_(not prob.has_hessians())
        self.assert_(not prob.has_gradient_sparsity())
        self.assert_(not prob.has_hessians_sparsity())
        self.assert_(not prob.is_stochastic())
        self.assert_(prob.is_(p))
        self.assert_(not prob.is_(int))
        self.assert_(id(prob.extract(p)) != id(p_inst))
        self.assert_(prob.extract(int) is None)
        # Fitness.
        self.assert_(all(prob.fitness([0, 0]) == [42]))
        # Run fitness a few more times.
        prob.fitness([0, 0])
        prob.fitness([0, 0])
        # Assert that the global variable was copied into p, not simply
        # referenced.
        self.assertEqual(len(glob), 0)
        self.assertEqual(len(prob.extract(p).g), 3)
        # Non-finite bounds.

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, float('inf')])

            def fitness(self, a):
                return [42]
        prob = problem(p())
        self.assert_(all(prob.get_bounds()[0] == [0, 0]))
        self.assert_(all(prob.get_bounds()[1] == [1, float('inf')]))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, float('nan')])

            def fitness(self, a):
                return [42]
        self.assertRaises(ValueError, lambda: problem(p()))
        # Wrong bounds.

        class p(object):

            def get_bounds(self):
                return ([0, 0], [-1, -1])

            def fitness(self, a):
                return [42]
        self.assertRaises(ValueError, lambda: problem(p()))
        # Wrong bounds type.

        class p(object):

            def get_bounds(self):
                return [[0, 0], [-1, -1]]

            def fitness(self, a):
                return [42]
        self.assertRaises(TypeError, lambda: problem(p()))
        # Bounds returned as numpy arrays.

        class p(object):

            def get_bounds(self):
                return (array([0., 0.]), array([1, 1]))

            def fitness(self, a):
                return [42]
        prob = problem(p())
        self.assert_(all(prob.get_bounds()[0] == [0, 0]))
        self.assert_(all(prob.get_bounds()[1] == [1, 1]))
        # Bounds returned as mixed types.

        class p(object):

            def get_bounds(self):
                return ([0., 1], (2., 3.))

            def fitness(self, a):
                return [42]
        prob = problem(p())
        self.assert_(all(prob.get_bounds()[0] == [0, 1]))
        self.assert_(all(prob.get_bounds()[1] == [2, 3]))
        # Invalid fitness size.

        class p(object):

            def get_bounds(self):
                return (array([0., 0.]), array([1, 1]))

            def fitness(self, a):
                assert(type(a) == type(array([1.])))
                return [42, 43]
        prob = problem(p())
        self.assertRaises(ValueError, lambda: prob.fitness([1, 2]))
        # Invalid fitness dimensions.

        class p(object):

            def get_bounds(self):
                return (array([0., 0.]), array([1, 1]))

            def fitness(self, a):
                return array([[42], [43]])
        prob = problem(p())
        self.assertRaises(ValueError, lambda: prob.fitness([1, 2]))
        # Invalid fitness type.

        class p(object):

            def get_bounds(self):
                return (array([0., 0.]), array([1, 1]))

            def fitness(self, a):
                return 42
        prob = problem(p())
        self.assertRaises(AttributeError, lambda: prob.fitness([1, 2]))
        # Fitness returned as array.

        class p(object):

            def get_bounds(self):
                return (array([0., 0.]), array([1, 1]))

            def fitness(self, a):
                return array([42])
        prob = problem(p())
        self.assert_(all(prob.fitness([1, 2]) == array([42])))
        # Fitness returned as tuple.

        class p(object):

            def get_bounds(self):
                return (array([0., 0.]), array([1, 1]))

            def fitness(self, a):
                return (42,)
        prob = problem(p())
        self.assert_(all(prob.fitness([1, 2]) == array([42])))

    def run_nobj_tests(self):
        from .core import problem

        class p(object):

            def get_nobj(self):
                return 2

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43]
        prob = problem(p())
        self.assertEqual(prob.get_nobj(), 2)
        # Wrong number of nobj.

        class p(object):

            def get_nobj(self):
                return 0

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43]
        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_nobj(self):
                return -1

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43]
        self.assertRaises(OverflowError, lambda: problem(p()))
        # Inconsistent nobj.

        class p(object):

            def get_nobj(self):
                return 2

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]
        prob = problem(p())
        self.assertRaises(ValueError, lambda: prob.fitness([1, 2]))

    def run_extract_tests(self):
        from .core import problem, translate, _test_problem
        import sys

        # First we try with a C++ test problem.
        p = problem(_test_problem())
        # Verify the refcount of p is increased after extract().
        rc = sys.getrefcount(p)
        tprob = p.extract(_test_problem)
        self.assert_(sys.getrefcount(p) == rc + 1)
        del tprob
        self.assert_(sys.getrefcount(p) == rc)
        # Verify we are modifying the inner object.
        p.extract(_test_problem).set_n(5)
        self.assert_(p.extract(_test_problem).get_n() == 5)
        # Chain extracts.
        t = translate(_test_problem(), [0])
        pt = problem(t)
        rc = sys.getrefcount(pt)
        tprob = pt.extract(translate)
        # Verify that extracrion of translate from the problem
        # increases the refecount of pt.
        self.assert_(sys.getrefcount(pt) == rc + 1)
        # Extract the _test_problem from translate.
        rc2 = sys.getrefcount(tprob)
        ttprob = tprob.extract(_test_problem)
        # The refcount of pt is not affected.
        self.assert_(sys.getrefcount(pt) == rc + 1)
        # The refcount of tprob has increased.
        self.assert_(sys.getrefcount(tprob) == rc2 + 1)
        del tprob
        # We can still access ttprob.
        self.assert_(ttprob.get_n() == 1)
        self.assert_(sys.getrefcount(pt) == rc + 1)
        del ttprob
        # Now the refcount of pt decreases, because deleting
        # ttprob eliminates the last ref to tprob, which in turn
        # decreases the refcount of pt.
        self.assert_(sys.getrefcount(pt) == rc)

        class tproblem(object):

            def __init__(self):
                self._n = 1

            def get_n(self):
                return self._n

            def set_n(self, n):
                self._n = n

            def fitness(self, dv):
                return [0]

            def get_bounds(self):
                return ([0], [1])

        # Test with Python problem.
        p = problem(tproblem())
        rc = sys.getrefcount(p)
        tprob = p.extract(tproblem)
        # Reference count does not increase because
        # tproblem is stored as a proper Python object
        # with its own refcount.
        self.assert_(sys.getrefcount(p) == rc)
        self.assert_(tprob.get_n() == 1)
        tprob.set_n(12)
        self.assert_(p.extract(tproblem).get_n() == 12)

    def run_nec_nic_tests(self):
        from .core import problem

        class p(object):

            def get_nec(self):
                return 2

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]
        prob = problem(p())
        self.assertEqual(prob.get_nf(), 3)

        class p(object):

            def get_nec(self):
                return -1

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]
        self.assertRaises(OverflowError, lambda: problem(p()))

        class p(object):

            def get_nic(self):
                return 2

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]
        prob = problem(p())
        self.assertEqual(prob.get_nf(), 3)

        class p(object):

            def get_nic(self):
                return -1

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]
        self.assertRaises(OverflowError, lambda: problem(p()))

        class p(object):

            def get_nec(self):
                return 2

            def get_nic(self):
                return 3

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]
        prob = problem(p())
        self.assertEqual(prob.get_nf(), 6)

    def run_has_gradient_tests(self):
        from .core import problem

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        self.assert_(not problem(p()).has_gradient())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def has_gradient(self):
                return True

        self.assert_(not problem(p()).has_gradient())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient(self, dv):
                return [0]

            def has_gradient(self):
                return False

        self.assert_(not problem(p()).has_gradient())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient(self, dv):
                return [0]

        self.assert_(problem(p()).has_gradient())

    def run_gradient_tests(self):
        from numpy import array
        from .core import problem

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        self.assertRaises(NotImplementedError,
                          lambda: problem(p()).gradient([1, 2]))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient(self, a):
                return [0]

        self.assertRaises(ValueError, lambda: problem(p()).gradient([1, 2]))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient(self, a):
                return (0, 1)

        self.assert_(all(array([0., 1.]) == problem(p()).gradient([1, 2])))
        self.assertRaises(ValueError, lambda: problem(p()).gradient([1]))


class pso_test_case(_ut.TestCase):
    """Test case for the UDA pso

    """

    def runTest(self):
        from .core import pso
        uda = pso()
        log = uda.get_log()
        seed = uda.get_seed()


class sa_test_case(_ut.TestCase):
    """Test case for the UDA simulated annealing

    """

    def runTest(self):
        from .core import simulated_annealing
        uda = simulated_annealing()
        log = uda.get_log()
        seed = uda.get_seed()


class compass_search_test_case(_ut.TestCase):
    """Test case for the UDA compass search

    """

    def runTest(self):
        from .core import compass_search
        uda = compass_search()
        log = uda.get_log()


class cmaes_test_case(_ut.TestCase):
    """Test case for the UDA cmaes

    """

    def runTest(self):
        from .core import cmaes
        uda = cmaes()
        log = uda.get_log()
        seed = uda.get_seed()


def run_test_suite():
    """Run the full test suite.

    This function will raise an exception if at least one test fails.

    """
    retval = 0
    suite = _ut.TestLoader().loadTestsFromTestCase(core_test_case)
    suite.addTest(problem_test_case())
    suite.addTest(pso_test_case())
    suite.addTest(cmaes_test_case())
    suite.addTest(compass_search_test_case())
    suite.addTest(sa_test_case())
    test_result = _ut.TextTestRunner(verbosity=2).run(suite)
    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1
    if retval != 0:
        raise RuntimeError('One or more tests failed.')
