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


class population_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.core.population` class.

    """

    def runTest(self):
        self.run_champion_test()

    def run_champion_test(self):
        from .core import population, null_problem, problem
        from numpy import array
        udp = null_problem()
        prob = problem(udp)
        pop = population(prob)
        self.assertEqual(len(pop.champion_f), 0)
        self.assertEqual(len(pop.champion_x), 0)
        pop.push_back([1.])
        self.assertEqual(pop.champion_f[0], 0.)
        self.assertEqual(pop.champion_x[0], 1.)


class pso_test_case(_ut.TestCase):
    """Test case for the UDA pso

    """

    def runTest(self):
        from .core import pso
        uda = pso()
        uda = pso(gen=1, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5,
                  variant=5, neighb_type=2, neighb_param=4, memory=False)
        uda = pso(gen=1, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5,
                  variant=5, neighb_type=2, neighb_param=4, memory=False, seed=32)
        self.assertEqual(uda.get_seed(), 32)
        log = uda.get_log()


class sa_test_case(_ut.TestCase):
    """Test case for the UDA simulated annealing

    """

    def runTest(self):
        from .core import simulated_annealing
        uda = simulated_annealing()
        uda = simulated_annealing(
            Ts=10., Tf=.1, n_T_adj=10, n_range_adj=10, bin_size=10, start_range=1.)
        uda = simulated_annealing(
            Ts=10., Tf=.1, n_T_adj=10, n_range_adj=10, bin_size=10, start_range=1., seed=32)
        log = uda.get_log()
        self.assertEqual(uda.get_seed(), 32)
        seed = uda.get_seed()


class compass_search_test_case(_ut.TestCase):
    """Test case for the UDA compass search

    """

    def runTest(self):
        from .core import compass_search
        uda = compass_search()
        uda = compass_search(max_fevals=1, start_range=.1,
                             stop_range=.01, reduction_coeff=.5)
        log = uda.get_log()


class cmaes_test_case(_ut.TestCase):
    """Test case for the UDA cmaes

    """

    def runTest(self):
        from .core import cmaes
        uda = cmaes()
        uda = cmaes(gen=1, cc=-1, cs=-1, c1=-1, cmu=-1,
                    sigma0=0.5, ftol=1e-6, xtol=1e-6, memory=False)
        uda = cmaes(gen=1, cc=-1, cs=-1, c1=-1, cmu=-1, sigma0=0.5,
                    ftol=1e-6, xtol=1e-6, memory=False, seed=32)
        self.assertEqual(uda.get_seed(), 32)
        seed = uda.get_seed()


class null_problem_test_case(_ut.TestCase):
    """Test case for the null problem

    """

    def runTest(self):
        from .core import null_problem as np, problem
        n = np()
        n = np(1)
        n = np(nobj=2)
        self.assertRaises(ValueError, lambda: np(0))
        self.assertTrue(problem(np()).get_nobj() == 1)
        self.assertTrue(problem(np(23)).get_nobj() == 23)


class hypervolume_test_case(_ut.TestCase):
    """Test case for the hypervolume utilities

    """

    def runTest(self):
        from .core import hypervolume, hv2d, hv3d, hvwfg, bf_fpras, bf_approx
        from .core import population, zdt
        pop = population(prob=zdt(id=1, param=10), size=20)
        hv1 = hypervolume(pop=pop)
        hv2 = hypervolume(points=[[0, 0], [-1, 1], [-2, 2]])
        hv2.copy_points = True
        points = hv2.get_points()
        res0 = hv2.compute([3, 3])

        algo1 = hv2d()
        algo2 = hvwfg()
        algo3 = bf_fpras()
        algo4 = bf_approx()

        res = hv2.compute(ref_point=[3, 3], hv_algo=algo1)
        res = hv2.exclusive(idx=0, ref_point=[3, 3], hv_algo=algo1)
        res = hv2.least_contributor(ref_point=[3, 3], hv_algo=algo1)
        res = hv2.greatest_contributor(ref_point=[3, 3], hv_algo=algo1)
        res = hv2.contributions(ref_point=[3, 3], hv_algo=algo1)
        res = hv2.compute(ref_point=[3, 3], hv_algo=algo2)
        res = hv2.exclusive(idx=0, ref_point=[3, 3], hv_algo=algo2)
        res = hv2.least_contributor(ref_point=[3, 3], hv_algo=algo2)
        res = hv2.greatest_contributor(ref_point=[3, 3], hv_algo=algo2)
        res = hv2.contributions(ref_point=[3, 3], hv_algo=algo2)
        res = hv2.compute(ref_point=[3, 3], hv_algo=algo3)

        res = hv2.least_contributor(ref_point=[3, 3], hv_algo=algo4)
        res = hv2.greatest_contributor(ref_point=[3, 3], hv_algo=algo4)

        res = hv2.compute(ref_point=[3, 3])
        res = hv2.exclusive(idx=0, ref_point=[3, 3])
        res = hv2.least_contributor(ref_point=[3, 3])
        res = hv2.greatest_contributor(ref_point=[3, 3])
        res = hv2.contributions(ref_point=[3, 3])


class dtlz_test_case(_ut.TestCase):
    """Test case for the UDP dtlz

    """

    def runTest(self):
        from .core import dtlz, population
        udp = dtlz(id=3, dim=9, fdim=3, alpha=5)
        udp.p_distance([0.2] * 9)
        udp.p_distance(population(udp, 20))


class translate_test_case(_ut.TestCase):
    """Test case for the translate meta-problem

    """

    def runTest(self):
        from .core import problem, rosenbrock, translate, null_problem, decompose
        from numpy import array

        t = translate()
        self.assertFalse(t.extract(null_problem) is None)
        self.assertTrue(all(t.translation == array([0.])))
        t = translate(udp=rosenbrock(), translation=[1, 2])
        self.assertFalse(t.extract(rosenbrock) is None)
        self.assertTrue(all(t.translation == array([1., 2.])))
        t = translate(rosenbrock(), [1, 2])
        self.assertTrue(problem(t).is_(translate))
        self.assertFalse(problem(t).extract(translate) is None)
        self.assertTrue(t.is_(rosenbrock))
        self.assertFalse(t.extract(rosenbrock) is None)
        self.assertTrue(all(t.translation == array([1., 2.])))
        t = translate(translation=[1, 2], udp=rosenbrock())
        self.assertFalse(t.extract(rosenbrock) is None)
        self.assertTrue(all(t.translation == array([1., 2.])))

        # Nested translation.
        t = translate(translate(rosenbrock(), [1, 2]), [1, 2])
        self.assertTrue(t.is_(translate))
        self.assertFalse(t.extract(translate) is None)
        self.assertFalse(t.extract(translate).extract(rosenbrock) is None)

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        t = translate(p(), [-1, -1])
        self.assertFalse(t.extract(p) is None)
        self.assertTrue(all(t.translation == array([-1., -1.])))
        t = translate(translation=[-1, -1], udp=p())
        self.assertTrue(t.is_(p))
        self.assertFalse(t.extract(p) is None)
        self.assertTrue(all(t.translation == array([-1., -1.])))

        # Verify construction from problem is forbidden.
        self.assertRaises(TypeError, lambda: translate(
            problem(null_problem()), [0.]))

        # Verify translation of decompose.
        t = translate(decompose(null_problem(2), [0.2, 0.8], [0., 0.]), [0.])
        self.assertTrue(t.is_(decompose))
        self.assertFalse(t.extract(decompose) is None)
        self.assertFalse(t.extract(decompose).extract(null_problem) is None)


class decompose_test_case(_ut.TestCase):
    """Test case for the decompose meta-problem

    """

    def runTest(self):
        from .core import zdt, decompose, null_problem, problem, translate
        from numpy import array

        d = decompose()
        self.assertFalse(d.extract(null_problem) is None)
        self.assertTrue(all(d.z == array([0., 0.])))
        d = decompose(zdt(1, 2), [0.5, 0.5], [0.1, 0.1], "weighted", False)
        self.assertTrue(problem(d).is_(decompose))
        self.assertFalse(problem(d).extract(decompose) is None)
        self.assertTrue(d.is_(zdt))
        self.assertFalse(d.extract(zdt) is None)
        self.assertTrue(all(d.z == array([0.1, 0.1])))
        self.assertTrue(all(d.original_fitness(
            [1., 1.]) == problem(zdt(1, 2)).fitness([1., 1.])))
        f = problem(zdt(1, 2)).fitness([1., 1.])
        fdw = d.decompose_fitness(f, [0.2, 0.8], [0.1, 0.1])

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43]

            def get_nobj(self):
                return 2

        d = decompose(p(), [0.5, 0.5], [0.1, 0.1], "weighted", False)
        self.assertTrue(d.is_(p))
        self.assertFalse(d.extract(p) is None)
        self.assertTrue(all(d.z == array([0.1, 0.1])))
        self.assertTrue(all(d.original_fitness([1., 1.]) == array([42, 43])))
        d.decompose_fitness([42, 43], [0.2, 0.8], [0.1, 0.1])

        # Verify construction from problem is forbidden.
        self.assertRaises(TypeError, lambda: decompose(
            problem(null_problem(2)), [0.2, 0.8], [0., 0.]))

        # Verify decomposition of translate.
        t = decompose(translate(null_problem(2), [0.]), [0.2, 0.8], [0., 0.])
        self.assertTrue(t.is_(translate))
        self.assertFalse(t.extract(translate) is None)
        self.assertFalse(t.extract(translate).extract(null_problem) is None)


def run_test_suite():
    """Run the full test suite.

    This function will raise an exception if at least one test fails.

    """
    from . import _problem_test, _algorithm_test
    retval = 0
    suite = _ut.TestLoader().loadTestsFromTestCase(core_test_case)
    suite.addTest(_problem_test.problem_test_case())
    suite.addTest(_algorithm_test.algorithm_test_case())
    suite.addTest(pso_test_case())
    suite.addTest(compass_search_test_case())
    suite.addTest(sa_test_case())
    suite.addTest(population_test_case())
    suite.addTest(null_problem_test_case())
    suite.addTest(hypervolume_test_case())
    try:
        from .core import cmaes
        suite.addTest(cmaes_test_case())
    except ImportError:
        pass
    suite.addTest(dtlz_test_case())
    suite.addTest(translate_test_case())
    suite.addTest(decompose_test_case())
    test_result = _ut.TextTestRunner(verbosity=2).run(suite)
    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1
    if retval != 0:
        raise RuntimeError('One or more tests failed.')
