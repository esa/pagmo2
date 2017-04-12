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


class _prob(object):

    def get_bounds(self):
        return ([0, 0], [1, 1])

    def fitness(self, a):
        return [42]


class problem_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.problem` class.

    """

    def runTest(self):
        self.run_basic_tests()
        self.run_extract_tests()
        self.run_nobj_tests()
        self.run_nec_nic_tests()
        self.run_nc_tests()
        self.run_nx_tests()
        self.run_nf_tests()
        self.run_ctol_tests()
        self.run_evals_tests()
        self.run_has_gradient_tests()
        self.run_gradient_tests()
        self.run_has_gradient_sparsity_tests()
        self.run_gradient_sparsity_tests()
        self.run_has_hessians_tests()
        self.run_hessians_tests()
        self.run_has_hessians_sparsity_tests()
        self.run_hessians_sparsity_tests()
        self.run_seed_tests()
        self.run_feas_tests()
        self.run_name_info_tests()
        self.run_thread_safety_tests()
        self.run_pickle_test()

    def run_basic_tests(self):
        # Tests for minimal problem, and mandatory methods.
        from numpy import all, array
        from .core import problem, rosenbrock, null_problem
        # Def construction.
        p = problem()
        self.assertTrue(p.extract(null_problem) is not None)
        self.assertTrue(p.extract(rosenbrock) is None)
        self.assertEqual(p.get_nobj(), 1)
        self.assertEqual(p.get_nx(), 1)

        # First a few non-problems.
        self.assertRaises(NotImplementedError, lambda: problem(1))
        self.assertRaises(NotImplementedError, lambda: problem("hello world"))
        self.assertRaises(NotImplementedError, lambda: problem([]))
        self.assertRaises(TypeError, lambda: problem(int))
        # Some problems missing methods, wrong arity, etc.

        class np0(object):

            def fitness(self, a):
                return [1]
        self.assertRaises(NotImplementedError, lambda: problem(np0()))

        class np1(object):

            def get_bounds(self):
                return ([0], [1])
        self.assertRaises(NotImplementedError, lambda: problem(np1()))

        class np2(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            fitness = 42
        self.assertRaises(NotImplementedError, lambda: problem(np2()))

        class np3(object):

            def get_bounds(self, a):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]
        self.assertRaises(TypeError, lambda: problem(np3()))
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
        # Test the keyword arg.
        prob = problem(udp=rosenbrock())
        prob = problem(udp=p_inst)
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

    def run_ctol_tests(self):
        from .core import problem
        from numpy import array

        class p(object):

            def get_nobj(self):
                return 2

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43]
        prob = problem(p())
        self.assertTrue(all(prob.c_tol == array([])))

        class p(object):

            def get_nobj(self):
                return 2

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43, 44]

            def get_nec(self):
                return 1
        prob = problem(p())
        self.assertTrue(all(prob.c_tol == array([0.])))

        class p(object):

            def get_nobj(self):
                return 2

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43, 44, 45]

            def get_nec(self):
                return 1

            def get_nic(self):
                return 1
        prob = problem(p())
        self.assertTrue(all(prob.c_tol == array([0., 0.])))

        def raiser():
            prob.c_tol = []
        self.assertRaises(ValueError, raiser)
        self.assertTrue(all(prob.c_tol == array([0., 0.])))

        def raiser():
            prob.c_tol = [1, 2, 3]
        self.assertRaises(ValueError, raiser)
        self.assertTrue(all(prob.c_tol == array([0., 0.])))

        def raiser():
            prob.c_tol = [1., float("NaN")]
        self.assertRaises(ValueError, raiser)
        self.assertTrue(all(prob.c_tol == array([0., 0.])))

        def raiser():
            prob.c_tol = [1., -1.]
        self.assertRaises(ValueError, raiser)
        self.assertTrue(all(prob.c_tol == array([0., 0.])))
        prob.c_tol = [1e-8, 1e-6]
        self.assertTrue(all(prob.c_tol == array([1e-8, 1e-6])))

    def run_evals_tests(self):
        from .core import problem
        from numpy import array

        class p(object):

            def get_nobj(self):
                return 2

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43]

            def gradient(self, a):
                return [1, 2, 3, 4]

            def hessians(self, a):
                return [[1, 2, 3], [4, 5, 6]]
        prob = problem(p())
        self.assertEqual(prob.get_fevals(), 0)
        self.assertEqual(prob.get_gevals(), 0)
        self.assertEqual(prob.get_hevals(), 0)
        prob.fitness([1, 2])
        self.assertEqual(prob.get_fevals(), 1)
        prob.gradient([1, 2])
        self.assertEqual(prob.get_gevals(), 1)
        prob.hessians([1, 2])
        self.assertEqual(prob.get_hevals(), 1)

    def run_nx_tests(self):
        from .core import problem

        class p(object):

            def get_nobj(self):
                return 2

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43]
        prob = problem(p())
        self.assertEqual(prob.get_nx(), 2)

        class p(object):

            def get_nobj(self):
                return 2

            def get_bounds(self):
                return ([0, 0, 1], [1, 1, 2])

            def fitness(self, a):
                return [42, 43]
        prob = problem(p())
        self.assertEqual(prob.get_nx(), 3)

    def run_nf_tests(self):
        from .core import problem

        class p(object):

            def get_nobj(self):
                return 2

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43]
        prob = problem(p())
        self.assertEqual(prob.get_nf(), 2)

        class p(object):

            def get_nobj(self):
                return 2

            def get_nec(self):
                return 1

            def get_bounds(self):
                return ([0, 0, 1], [1, 1, 2])

            def fitness(self, a):
                return [42, 43, 44]
        prob = problem(p())
        self.assertEqual(prob.get_nf(), 3)

        class p(object):

            def get_nobj(self):
                return 2

            def get_nic(self):
                return 1

            def get_bounds(self):
                return ([0, 0, 1], [1, 1, 2])

            def fitness(self, a):
                return [42, 43, 44]
        prob = problem(p())
        self.assertEqual(prob.get_nf(), 3)

        class p(object):

            def get_nobj(self):
                return 2

            def get_nic(self):
                return 1

            def get_nec(self):
                return 2

            def get_bounds(self):
                return ([0, 0, 1], [1, 1, 2])

            def fitness(self, a):
                return [42, 43, 44]
        prob = problem(p())
        self.assertEqual(prob.get_nf(), 5)

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
        from .core import problem, translate, _test_problem, decompose
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
        # Verify that extraction of translate from the problem
        # increases the refecount of pt.
        self.assert_(sys.getrefcount(pt) == rc + 1)
        # Get back the _test_problem from translate.
        rc2 = sys.getrefcount(tprob)
        ttprob = tprob.inner_problem.extract(_test_problem)
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

        # Do the same with decompose.
        p = problem(_test_problem(2))
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
        t = decompose(_test_problem(2), [.2, .8], [0., 0.])
        pt = problem(t)
        rc = sys.getrefcount(pt)
        tprob = pt.extract(decompose)
        # Verify that extraction of decompose from the problem
        # increases the refecount of pt.
        self.assert_(sys.getrefcount(pt) == rc + 1)
        # Extract the _test_problem from decompose.
        rc2 = sys.getrefcount(tprob)
        ttprob = tprob.inner_problem.extract(_test_problem)
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

        # Try chaining decompose and translate.
        p = problem(
            translate(decompose(_test_problem(2), [.2, .8], [0., 0.]), [1.]))
        rc = sys.getrefcount(p)
        tprob = p.extract(translate)
        self.assertFalse(tprob is None)
        self.assert_(sys.getrefcount(p) == rc + 1)
        tmp = sys.getrefcount(tprob)
        dprob = tprob.inner_problem.extract(decompose)
        self.assertFalse(dprob is None)
        self.assert_(sys.getrefcount(tprob) == tmp + 1)
        self.assert_(sys.getrefcount(p) == rc + 1)
        tmp2 = sys.getrefcount(dprob)
        test_prob = dprob.inner_problem.extract(_test_problem)
        self.assertFalse(test_prob is None)
        self.assert_(sys.getrefcount(dprob) == tmp2 + 1)
        self.assert_(sys.getrefcount(p) == rc + 1)
        del tprob
        # We can still access dprob and test_prob.
        dprob.z
        self.assertTrue(test_prob.get_n() == 1)
        del dprob
        del test_prob
        # Verify the refcount of p drops back.
        self.assert_(sys.getrefcount(p) == rc)

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

    def run_nc_tests(self):
        from .core import problem

        class p(object):

            def get_nec(self):
                return 2

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]
        prob = problem(p())
        self.assertEqual(prob.get_nc(), 2)

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
        self.assertEqual(prob.get_nc(), 5)

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]
        prob = problem(p())
        self.assertEqual(prob.get_nc(), 0)

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

    def run_has_gradient_sparsity_tests(self):
        from .core import problem

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        self.assert_(not problem(p()).has_gradient_sparsity())

        class p(object):

            def get_bounds(self):
                return ([0], [1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return [(0, 0)]

        self.assert_(problem(p()).has_gradient_sparsity())

        class p(object):

            def get_bounds(self):
                return ([0], [1])

            def fitness(self, a):
                return [42]

            def has_gradient_sparsity(self):
                return True

        self.assert_(not problem(p()).has_gradient_sparsity())

        class p(object):

            def get_bounds(self):
                return ([0], [1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return [(0, 0)]

            def has_gradient_sparsity(self):
                return True

        self.assert_(problem(p()).has_gradient_sparsity())

        class p(object):

            def get_bounds(self):
                return ([0], [1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return [(0, 0)]

            def has_gradient_sparsity(self):
                return False

        self.assert_(not problem(p()).has_gradient_sparsity())

    def run_gradient_sparsity_tests(self):
        from .core import problem
        from numpy import array, ndarray

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return ()

        self.assert_(problem(p()).has_gradient_sparsity())
        self.assert_(isinstance(problem(p()).gradient_sparsity(), ndarray))
        self.assert_(problem(p()).gradient_sparsity().shape == (0, 2))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return []

        self.assert_(problem(p()).has_gradient_sparsity())
        self.assert_(isinstance(problem(p()).gradient_sparsity(), ndarray))
        self.assert_(problem(p()).gradient_sparsity().shape == (0, 2))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return {}

        self.assert_(problem(p()).has_gradient_sparsity())
        self.assert_(isinstance(problem(p()).gradient_sparsity(), ndarray))
        self.assert_(problem(p()).gradient_sparsity().shape == (0, 2))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return [[0, 0]]

        self.assert_(problem(p()).has_gradient_sparsity())
        self.assert_(isinstance(problem(p()).gradient_sparsity(), ndarray))
        self.assert_(problem(p()).gradient_sparsity().shape == (1, 2))
        self.assert_((problem(p()).gradient_sparsity()
                      == array([[0, 0]])).all())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return [[0, 0], (0, 1)]

        self.assert_(problem(p()).has_gradient_sparsity())
        self.assert_(isinstance(problem(p()).gradient_sparsity(), ndarray))
        self.assert_(problem(p()).gradient_sparsity().shape == (2, 2))
        self.assert_((problem(p()).gradient_sparsity()
                      == array([[0, 0], [0, 1]])).all())
        self.assertEqual(problem(p()).gradient_sparsity()[0][0], 0)
        self.assertEqual(problem(p()).gradient_sparsity()[0][1], 0)
        self.assertEqual(problem(p()).gradient_sparsity()[1][0], 0)
        self.assertEqual(problem(p()).gradient_sparsity()[1][1], 1)

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return [[0, 0], (0,)]

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return [[0, 0], (0, 0)]

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return [[0, 0], (0, 123)]

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return array([[0, 0], [0, 1]])

        self.assert_(problem(p()).has_gradient_sparsity())
        self.assert_(isinstance(problem(p()).gradient_sparsity(), ndarray))
        self.assert_(problem(p()).gradient_sparsity().shape == (2, 2))
        self.assert_((problem(p()).gradient_sparsity()
                      == array([[0, 0], [0, 1]])).all())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return array([[0, 0], [0, 123]])

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return array([[0, 0, 0], [0, 1, 0]])

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return array([[[0], [1], [2]]])

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return [[[0], 0], [0, 1]]

        self.assertRaises(TypeError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return array([[0, 0], [0, -1]])

        self.assertRaises(OverflowError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                a = array([[0, 0, 0], [0, 1, 0]])
                return a[:, :2]

        self.assert_(problem(p()).has_gradient_sparsity())
        self.assert_(isinstance(problem(p()).gradient_sparsity(), ndarray))
        self.assert_(problem(p()).gradient_sparsity().shape == (2, 2))
        self.assert_((problem(p()).gradient_sparsity()
                      == array([[0, 0], [0, 1]])).all())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return array([[0, 0], [0, 1.]])

        self.assertRaises(TypeError, lambda: problem(p()))

        class p(object):
            counter = 0

            def get_bounds(self):
                return ([0], [1])

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                if p.counter == 0:
                    p.counter = p.counter + 1
                    return []
                return [(0, 0)]

        self.assertRaises(ValueError, lambda: problem(p()).gradient_sparsity())

        class p(object):

            def get_bounds(self):
                return ([0] * 6, [1] * 6)

            def fitness(self, a):
                return [42]

            def gradient_sparsity(self):
                return [(0, 0), (0, 2), (0, 1)]

        self.assertRaises(ValueError, lambda: problem(p()))

    def run_has_hessians_tests(self):
        from .core import problem

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        self.assert_(not problem(p()).has_hessians())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def has_hessians(self):
                return True

        self.assert_(not problem(p()).has_hessians())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians(self, dv):
                return [0]

            def has_hessians(self):
                return False

        self.assert_(not problem(p()).has_hessians())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians(self, dv):
                return [0]

        self.assert_(problem(p()).has_hessians())

    def run_hessians_tests(self):
        from numpy import array
        from .core import problem

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        self.assertRaises(NotImplementedError,
                          lambda: problem(p()).hessians([1, 2]))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians(self, a):
                return [0]

        # Rasies AttributeError because we are trying to iterate over
        # the element of the returned hessians, which, in this case, is
        # an int.
        self.assertRaises(
            AttributeError, lambda: problem(p()).hessians([1, 2]))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians(self, a):
                return [(1, 2, 3)]

        self.assert_(all(array([1., 2., 3.]) ==
                         problem(p()).hessians([1, 2])[0]))
        self.assertRaises(ValueError, lambda: problem(p()).hessians([1]))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians(self, a):
                return ([1, 2, 3],)

        self.assert_(all(array([1., 2., 3.]) ==
                         problem(p()).hessians([1, 2])[0]))
        self.assertRaises(ValueError, lambda: problem(p()).hessians([1]))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians(self, a):
                return (array([1, 2, 3]),)

        self.assert_(all(array([1., 2., 3.]) ==
                         problem(p()).hessians([1, 2])[0]))
        self.assertRaises(ValueError, lambda: problem(p()).hessians([1]))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, -42]

            def get_nobj(self):
                return 2

            def hessians(self, a):
                return (array([1, 2, 3]), (4, 5, 6))

        self.assert_(all(array([1., 2., 3.]) ==
                         problem(p()).hessians([1, 2])[0]))
        self.assert_(all(array([4., 5., 6.]) ==
                         problem(p()).hessians([1, 2])[1]))
        self.assertRaises(ValueError, lambda: problem(p()).hessians([1]))

        class p(object):

            def get_bounds(self):
                return ([0] * 6, [1] * 6)

            def fitness(self, a):
                return [42, -42]

            def get_nobj(self):
                return 2

            def hessians(self, a):
                return []

        self.assertRaises(ValueError, lambda: problem(p()).hessians([1] * 6))

    def run_has_hessians_sparsity_tests(self):
        from .core import problem

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        self.assert_(not problem(p()).has_hessians_sparsity())

        class p(object):

            def get_bounds(self):
                return ([0], [1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return [[(0, 0)]]

        self.assert_(problem(p()).has_hessians_sparsity())

        class p(object):

            def get_bounds(self):
                return ([0], [1])

            def fitness(self, a):
                return [42]

            def has_hessians_sparsity(self):
                return True

        self.assert_(not problem(p()).has_hessians_sparsity())

        class p(object):

            def get_bounds(self):
                return ([0], [1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return ([(0, 0)],)

            def has_hessians_sparsity(self):
                return True

        self.assert_(problem(p()).has_hessians_sparsity())

        class p(object):

            def get_bounds(self):
                return ([0], [1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return [array([[0, 0]])]

            def has_hessians_sparsity(self):
                return False

        self.assert_(not problem(p()).has_hessians_sparsity())

    def run_hessians_sparsity_tests(self):
        from .core import problem
        from numpy import array, ndarray

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return ([],)

        self.assert_(problem(p()).has_hessians_sparsity())
        self.assert_(isinstance(problem(p()).hessians_sparsity(), list))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return ([],)

        self.assert_(problem(p()).has_hessians_sparsity())
        self.assert_(isinstance(problem(p()).hessians_sparsity(), list))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return {()}

        self.assert_(problem(p()).has_hessians_sparsity())
        self.assert_(isinstance(problem(p()).hessians_sparsity(), list))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return [[(0, 0)]]

        self.assert_(problem(p()).has_hessians_sparsity())
        self.assert_(isinstance(problem(p()).hessians_sparsity(), list))
        self.assert_(isinstance(problem(p()).hessians_sparsity()[0], ndarray))
        self.assert_(problem(p()).hessians_sparsity()[0].shape == (1, 2))
        self.assert_((problem(p()).hessians_sparsity()[0]
                      == array([[0, 0]])).all())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return [[[0, 0], (1, 0)]]

        self.assert_(problem(p()).has_hessians_sparsity())
        self.assert_(isinstance(problem(p()).hessians_sparsity(), list))
        self.assert_(isinstance(problem(p()).hessians_sparsity()[0], ndarray))
        self.assert_(problem(p()).hessians_sparsity()[0].shape == (2, 2))
        self.assert_((problem(p()).hessians_sparsity()[0]
                      == array([[0, 0], [1, 0]])).all())
        self.assertEqual(problem(p()).hessians_sparsity()[0][0][0], 0)
        self.assertEqual(problem(p()).hessians_sparsity()[0][0][1], 0)
        self.assertEqual(problem(p()).hessians_sparsity()[0][1][1], 0)
        self.assertEqual(problem(p()).hessians_sparsity()[0][1][0], 1)

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return ([[0, 0], (0,)],)

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return [[[0, 0], (0, 0)]]

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return [[[0, 0], (0, 123)]]

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return [array([[0, 0], [1, 1]])]

        self.assert_(problem(p()).has_hessians_sparsity())
        self.assert_(isinstance(problem(p()).hessians_sparsity(), list))
        self.assert_(isinstance(problem(p()).hessians_sparsity()[0], ndarray))
        self.assert_(problem(p()).hessians_sparsity()[0].shape == (2, 2))
        self.assert_((problem(p()).hessians_sparsity()[0]
                      == array([[0, 0], [1, 1]])).all())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return array([[0, 0], [0, 123]])

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return (array([[0, 0, 0], [0, 1, 0]]),)

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return [array([[[0], [1], [2]]])]

        self.assertRaises(ValueError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return [[[[0], 0], [0, 1]]]

        self.assertRaises(TypeError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return [array([[0, 0], [0, -1]])]

        self.assertRaises(OverflowError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                a = array([[0, 0, 0], [1, 1, 0]])
                return [a[:, :2]]

        self.assert_(problem(p()).has_hessians_sparsity())
        self.assert_(isinstance(problem(p()).hessians_sparsity(), list))
        self.assert_(isinstance(problem(p()).hessians_sparsity()[0], ndarray))
        self.assert_(problem(p()).hessians_sparsity()[0].shape == (2, 2))
        self.assert_((problem(p()).hessians_sparsity()[0]
                      == array([[0, 0], [1, 1]])).all())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def hessians_sparsity(self):
                return [array([[0, 0], [0, 1.]])]

        self.assertRaises(TypeError, lambda: problem(p()))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43]

            def get_nobj(self):
                return 2

            def hessians_sparsity(self):
                return [array([[0, 0], [1, 1]]), array([[0, 0], [1, 0]])]

        self.assert_(problem(p()).has_hessians_sparsity())
        self.assert_(isinstance(problem(p()).hessians_sparsity(), list))
        self.assert_(isinstance(problem(p()).hessians_sparsity()[0], ndarray))
        self.assert_(isinstance(problem(p()).hessians_sparsity()[1], ndarray))
        self.assert_(problem(p()).hessians_sparsity()[0].shape == (2, 2))
        self.assert_(problem(p()).hessians_sparsity()[1].shape == (2, 2))
        self.assert_((problem(p()).hessians_sparsity()[0]
                      == array([[0, 0], [1, 1]])).all())
        self.assert_((problem(p()).hessians_sparsity()[1]
                      == array([[0, 0], [1, 0]])).all())

        class p(object):
            counter = 0

            def get_bounds(self):
                return ([0] * 6, [1] * 6)

            def fitness(self, a):
                return [42, 42]

            def get_nobj(self):
                return 2

            def hessians_sparsity(self):
                if p.counter == 0:
                    p.counter = p.counter + 1
                    return [[(1, 0)], [(1, 0)]]
                return [[(1, 0)], [(1, 0), (2, 0)]]

        self.assertRaises(ValueError, lambda: problem(p()).hessians_sparsity())

        class p(object):

            def get_bounds(self):
                return ([0] * 6, [1] * 6)

            def fitness(self, a):
                return [42, 42]

            def get_nobj(self):
                return 2

            def hessians_sparsity(self):
                return [[(1, 0)], [(1, 0), (2, 0), (1, 1)]]

        self.assertRaises(ValueError, lambda: problem(p()))

    def run_seed_tests(self):
        from .core import problem

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        self.assert_(not problem(p()).has_set_seed())
        self.assertRaises(NotImplementedError,
                          lambda: problem(p()).set_seed(12))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def has_set_seed(self):
                return True

        self.assert_(not problem(p()).has_set_seed())
        self.assertRaises(NotImplementedError,
                          lambda: problem(p()).set_seed(12))

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def set_seed(self, seed):
                pass

        self.assert_(problem(p()).has_set_seed())
        problem(p()).set_seed(87)

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def set_seed(self, seed):
                pass

            def has_set_seed(self):
                return False

        self.assert_(not problem(p()).has_set_seed())

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def set_seed(self, seed):
                pass

            def has_set_seed(self):
                return True

        self.assert_(problem(p()).has_set_seed())
        problem(p()).set_seed(0)
        problem(p()).set_seed(87)
        self.assertRaises(OverflowError, lambda: problem(p()).set_seed(-1))

    def run_feas_tests(self):
        from .core import problem

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        prob = problem(p())
        self.assertTrue(prob.feasibility_x([0, 0]))
        self.assertTrue(prob.feasibility_x(x=[0, 0]))
        self.assertEqual(2, prob.get_fevals())
        self.assertTrue(prob.feasibility_f([0]))
        self.assertTrue(prob.feasibility_f(f=[0]))
        self.assertEqual(2, prob.get_fevals())
        self.assertRaises(ValueError, lambda: prob.feasibility_f([0, 1]))

    def run_name_info_tests(self):
        from .core import problem

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        prob = problem(p())
        self.assert_(prob.get_name() != '')
        self.assert_(prob.get_extra_info() == '')

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def get_name(self):
                return 'pippo'

        prob = problem(p())
        self.assert_(prob.get_name() == 'pippo')
        self.assert_(prob.get_extra_info() == '')

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def get_extra_info(self):
                return 'pluto'

        prob = problem(p())
        self.assert_(prob.get_name() != '')
        self.assert_(prob.get_extra_info() == 'pluto')

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

            def get_name(self):
                return 'pippo'

            def get_extra_info(self):
                return 'pluto'

        prob = problem(p())
        self.assert_(prob.get_name() == 'pippo')
        self.assert_(prob.get_extra_info() == 'pluto')

    def run_thread_safety_tests(self):
        from .core import problem, rosenbrock, _tu_test_problem, translate
        from . import thread_safety as ts

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        self.assertTrue(problem(p()).get_thread_safety() == ts.none)
        self.assertTrue(problem(rosenbrock()).get_thread_safety() == ts.basic)
        self.assertTrue(
            problem(_tu_test_problem()).get_thread_safety() == ts.none)
        self.assertTrue(
            problem(translate(_tu_test_problem(), [0])).get_thread_safety() == ts.none)
        self.assertTrue(
            problem(translate(p(), [0, 1])).get_thread_safety() == ts.none)
        self.assertTrue(
            problem(translate(rosenbrock(), [0, 1])).get_thread_safety() == ts.basic)

    def run_pickle_test(self):
        from .core import problem, rosenbrock, translate
        from pickle import dumps, loads
        p = problem(rosenbrock(10))
        p = loads(dumps(p))
        self.assertEqual(repr(p), repr(problem(rosenbrock(10))))
        self.assertEqual(p.get_nobj(), 1)
        self.assertEqual(p.get_nx(), 10)
        self.assertTrue(p.is_(rosenbrock))
        p = problem(translate(rosenbrock(10), [.1] * 10))
        p = loads(dumps(p))
        self.assertEqual(repr(p), repr(
            problem(translate(rosenbrock(10), [.1] * 10))))
        self.assertEqual(p.get_nobj(), 1)
        self.assertEqual(p.get_nx(), 10)
        self.assertTrue(p.is_(translate))
        self.assertTrue(p.extract(translate).inner_problem.is_(rosenbrock))

        p = problem(_prob())
        p = loads(dumps(p))
        self.assertEqual(repr(p), repr(problem(_prob())))
        self.assertEqual(p.get_nobj(), 1)
        self.assertEqual(p.get_nx(), 2)
        self.assertTrue(p.is_(_prob))
        p = problem(translate(_prob(), [.1] * 2))
        p = loads(dumps(p))
        self.assertEqual(repr(p), repr(problem(translate(_prob(), [.1] * 2))))
        self.assertEqual(p.get_nobj(), 1)
        self.assertEqual(p.get_nx(), 2)
        self.assertTrue(p.is_(translate))
        self.assertTrue(p.extract(translate).inner_problem.is_(_prob))
