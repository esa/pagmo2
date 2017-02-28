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
    """Test case for the population class.

    """

    def runTest(self):
        self.run_init_test()
        self.run_best_worst_idx_test()
        self.run_champion_test()
        self.run_getters_test()
        self.run_problem_test()
        self.run_push_back_test()
        self.run_random_dv_test()
        self.run_set_x_xf_test()

    def run_init_test(self):
        from .core import population, null_problem, rosenbrock, problem
        pop = population()
        self.assertTrue(len(pop) == 0)
        self.assertTrue(pop.problem.extract(null_problem) is not None)
        self.assertTrue(pop.problem.extract(rosenbrock) is None)
        pop.get_seed()
        pop = population(rosenbrock())
        self.assertTrue(len(pop) == 0)
        self.assertTrue(pop.problem.extract(null_problem) is None)
        self.assertTrue(pop.problem.extract(rosenbrock) is not None)
        pop.get_seed()
        pop = population(seed=42, size=5, prob=problem(rosenbrock()))
        self.assertTrue(len(pop) == 5)
        self.assertTrue(pop.problem.extract(null_problem) is None)
        self.assertTrue(pop.problem.extract(rosenbrock) is not None)
        self.assertEqual(pop.get_seed(), 42)

    def run_best_worst_idx_test(self):
        from .core import population, rosenbrock, zdt
        pop = population(rosenbrock(), size=10)
        self.assertTrue(pop.best_idx() < 10)
        self.assertTrue(pop.best_idx(0.001) < 10)
        self.assertTrue(pop.best_idx(tol=0.001) < 10)
        self.assertTrue(pop.best_idx(tol=[0.001, 0.001]) < 10)
        self.assertTrue(pop.worst_idx() < 10)
        self.assertTrue(pop.worst_idx(0.001) < 10)
        self.assertTrue(pop.worst_idx(tol=0.001) < 10)
        self.assertTrue(pop.worst_idx(tol=[0.001, 0.001]) < 10)
        pop = population(zdt(param=10), size=10)
        self.assertRaises(ValueError, lambda: pop.best_idx())
        self.assertRaises(ValueError, lambda: pop.worst_idx())

    def run_champion_test(self):
        from .core import population, null_problem, problem, zdt
        from numpy import array
        udp = null_problem()
        prob = problem(udp)
        pop = population(prob)
        self.assertEqual(len(pop.champion_f), 0)
        self.assertEqual(len(pop.champion_x), 0)
        pop.push_back([1.])
        self.assertEqual(pop.champion_f[0], 0.)
        self.assertEqual(pop.champion_x[0], 1.)
        pop = population(zdt(param=10))
        self.assertRaises(ValueError, lambda: pop.champion_x)
        self.assertRaises(ValueError, lambda: pop.champion_f)

    def run_getters_test(self):
        from .core import population
        from numpy import ndarray
        pop = population(size=100, seed=123)
        self.assertEqual(len(pop.get_ID()), 100)
        self.assertTrue(isinstance(pop.get_ID(), ndarray))
        self.assertEqual(len(pop.get_f()), 100)
        self.assertTrue(isinstance(pop.get_f(), ndarray))
        self.assertEqual(pop.get_f().shape, (100, 1))
        self.assertEqual(len(pop.get_x()), 100)
        self.assertTrue(isinstance(pop.get_x(), ndarray))
        self.assertEqual(pop.get_x().shape, (100, 1))
        self.assertEqual(pop.get_seed(), 123)

    def run_problem_test(self):
        from .core import population, rosenbrock, null_problem, problem, zdt
        import sys
        pop = population(size=10)
        rc = sys.getrefcount(pop)
        prob = pop.problem
        self.assertTrue(sys.getrefcount(pop) == rc + 1)
        del prob
        self.assertTrue(sys.getrefcount(pop) == rc)
        self.assertTrue(pop.problem.extract(null_problem) is not None)
        self.assertTrue(pop.problem.extract(rosenbrock) is None)
        pop = population(rosenbrock(), size=10)
        self.assertTrue(pop.problem.extract(null_problem) is None)
        self.assertTrue(pop.problem.extract(rosenbrock) is not None)
        pop.problem = problem(zdt(param=10))
        self.assertRaises(ValueError, lambda: pop.best_idx())
        self.assertTrue(pop.problem.extract(null_problem) is None)
        self.assertTrue(pop.problem.extract(zdt) is not None)

    def run_push_back_test(self):
        from .core import population, rosenbrock
        from numpy import array
        pop = population(rosenbrock(), size=5)
        self.assertEqual(len(pop), 5)
        self.assertEqual(pop.problem.get_fevals(), 5)
        pop.push_back(x=[.1, .1])
        self.assertEqual(len(pop), 6)
        self.assertEqual(pop.problem.get_fevals(), 6)
        pop.push_back(x=[.1, .1], f=array([1]))
        self.assertEqual(len(pop), 7)
        self.assertEqual(pop.problem.get_fevals(), 6)
        pop.push_back(x=[.1, .1], f=array([0.0]))
        self.assertEqual(len(pop), 8)
        self.assertEqual(pop.problem.get_fevals(), 6)
        self.assertEqual(pop.best_idx(), 7)
        pop.push_back(x=[.1, .1], f=None)
        self.assertEqual(len(pop), 9)
        self.assertEqual(pop.problem.get_fevals(), 7)
        self.assertEqual(pop.best_idx(), 7)

    def run_random_dv_test(self):
        from .core import population, rosenbrock
        from numpy import ndarray
        pop = population(rosenbrock())
        self.assertTrue(isinstance(pop.random_decision_vector(), ndarray))
        self.assertTrue(pop.random_decision_vector().shape == (2,))
        self.assertTrue(pop.random_decision_vector()[0] >= -5)
        self.assertTrue(pop.random_decision_vector()[0] <= 10)
        self.assertTrue(pop.random_decision_vector()[1] >= -5)
        self.assertTrue(pop.random_decision_vector()[1] <= 10)

    def run_set_x_xf_test(self):
        from .core import population, rosenbrock
        from numpy import array
        pop = population(rosenbrock())
        self.assertRaises(ValueError, lambda: pop.set_x(0, [1, 1]))
        self.assertRaises(ValueError, lambda: pop.set_xf(0, (1, 1), [1]))
        pop = population(rosenbrock(), size=10)
        self.assertRaises(ValueError, lambda: pop.set_x(0, array([1, 1, 1])))
        self.assertRaises(ValueError, lambda: pop.set_xf(0, [1, 1], [1, 1]))
        self.assertRaises(ValueError, lambda: pop.set_xf(
            0, array([1, 1, 1]), [1, 1]))
        pop.set_x(0, array([1.1, 1.1]))
        self.assertTrue(all(pop.get_x()[0] == array([1.1, 1.1])))
        self.assertTrue(
            all(pop.get_f()[0] == pop.problem.fitness(array([1.1, 1.1]))))
        pop.set_x(4, array([1.1, 1.1]))
        self.assertTrue(all(pop.get_x()[4] == array([1.1, 1.1])))
        self.assertTrue(
            all(pop.get_f()[4] == pop.problem.fitness(array([1.1, 1.1]))))
        pop.set_xf(5, array([1.1, 1.1]), [1.25])
        self.assertTrue(all(pop.get_x()[5] == array([1.1, 1.1])))
        self.assertTrue(all(pop.get_f()[5] == array([1.25])))
        pop.set_xf(6, array([1.1, 1.1]), [0.])
        self.assertTrue(all(pop.get_x()[6] == array([1.1, 1.1])))
        self.assertTrue(all(pop.get_f()[6] == array([0])))
        self.assertEqual(pop.best_idx(), 6)


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


class moead_test_case(_ut.TestCase):
    """Test case for the UDA moead

    """

    def runTest(self):
        from .core import moead
        uda = moead()
        uda = moead(gen=1, weight_generation="grid", decomposition="tchebycheff",
                    neighbours=20, CR=1, F=0.5, eta_m=20, realb=0.9, limit=2, preserve_diversity=True)
        uda = moead(gen=1, weight_generation="grid", decomposition="tchebycheff", neighbours=20,
                    CR=1, F=0.5, eta_m=20, realb=0.9, limit=2, preserve_diversity=True, seed=32)
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


class nsga2_test_case(_ut.TestCase):
    """Test case for the UDA nsga2

    """

    def runTest(self):
        from .core import nsga2
        uda = nsga2()
        uda = nsga2(gen=1, cr=0.95, eta_c=10, m=0.01, eta_m=10, int_dim=0)
        uda = nsga2(gen=1, cr=0.95, eta_c=10, m=0.01,
                    eta_m=10, int_dim=0, seed=32)
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
        import numpy as np
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

        self.assertTrue((hv2.refpoint(offset=0) ==
                         np.array([0., 2.])).all() == True)
        self.assertTrue((hv2.refpoint(offset=.1) ==
                         np.array([0.1, 2.1])).all() == True)


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

        # Verify construction from problem is forbidden.
        self.assertRaises(TypeError, lambda: decompose(
            problem(null_problem(2)), [0.2, 0.8], [0., 0.]))

        # Verify decomposition of translate.
        t = decompose(translate(null_problem(2), [0.]), [0.2, 0.8], [0., 0.])
        self.assertTrue(t.is_(translate))
        self.assertFalse(t.extract(translate) is None)
        self.assertFalse(t.extract(translate).extract(null_problem) is None)


class mbh_test_case(_ut.TestCase):
    """Test case for the mbh meta-algorithm

    """

    def runTest(self):
        from . import mbh, de, compass_search, algorithm, thread_safety as ts, null_algorithm
        from numpy import array

        class algo(object):

            def evolve(pop):
                return pop

        # Def ctor.
        a = mbh()
        self.assertFalse(a.extract(compass_search) is None)
        self.assertTrue(a.is_(compass_search))
        self.assertTrue(a.extract(de) is None)
        self.assertFalse(a.is_(de))
        self.assertEqual(a.get_log(), [])
        self.assertTrue(all(a.get_perturb() == array([0.01])))
        seed = a.get_seed()
        self.assertEqual(a.get_verbosity(), 0)
        a.set_perturb([.2])
        self.assertTrue(all(a.get_perturb() == array([0.2])))
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.basic)
        self.assertTrue(al.extract(mbh).extract(compass_search) is not None)
        self.assertTrue(al.extract(mbh).extract(de) is None)
        self.assertTrue(str(seed) in str(al))
        al.set_verbosity(4)
        self.assertEqual(al.extract(mbh).get_verbosity(), 4)

        # From C++ algo.
        seed = 123321
        a = mbh(uda=de(), stop=5, perturb=.4)
        a = mbh(stop=5, perturb=(.4, .2), uda=de())
        a = mbh(uda=de(), stop=5, seed=seed, perturb=(.4, .2))
        self.assertTrue(a.extract(compass_search) is None)
        self.assertFalse(a.is_(compass_search))
        self.assertFalse(a.extract(de) is None)
        self.assertTrue(a.is_(de))
        self.assertEqual(a.get_log(), [])
        self.assertTrue(all(a.get_perturb() == array([.4, .2])))
        self.assertEqual(a.get_seed(), seed)
        self.assertEqual(a.get_verbosity(), 0)
        a.set_perturb([.2])
        self.assertTrue(all(a.get_perturb() == array([0.2])))
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.basic)
        self.assertTrue(al.extract(mbh).extract(compass_search) is None)
        self.assertTrue(al.extract(mbh).extract(de) is not None)
        self.assertTrue(str(seed) in str(al))
        al.set_verbosity(4)
        self.assertEqual(al.extract(mbh).get_verbosity(), 4)

        # From Python algo.
        class algo(object):

            def evolve(self, pop):
                return pop

        seed = 123321
        a = mbh(uda=algo(), stop=5, perturb=.4)
        a = mbh(stop=5, perturb=(.4, .2), uda=algo())
        a = mbh(uda=algo(), stop=5, seed=seed, perturb=(.4, .2))
        self.assertTrue(a.extract(compass_search) is None)
        self.assertFalse(a.is_(compass_search))
        self.assertFalse(a.extract(algo) is None)
        self.assertTrue(a.is_(algo))
        self.assertEqual(a.get_log(), [])
        self.assertTrue(all(a.get_perturb() == array([.4, .2])))
        self.assertEqual(a.get_seed(), seed)
        self.assertEqual(a.get_verbosity(), 0)
        a.set_perturb([.2])
        self.assertTrue(all(a.get_perturb() == array([0.2])))
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.none)
        self.assertTrue(al.extract(mbh).extract(compass_search) is None)
        self.assertTrue(al.extract(mbh).extract(algo) is not None)
        self.assertTrue(str(seed) in str(al))
        al.set_verbosity(4)
        self.assertEqual(al.extract(mbh).get_verbosity(), 4)

        # Construction from algorithm is forbidden.
        self.assertRaises(TypeError, lambda: mbh(
            algorithm(null_algorithm()), stop=5, perturb=.4))


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
    suite.addTest(moead_test_case())
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
    suite.addTest(mbh_test_case())
    test_result = _ut.TextTestRunner(verbosity=2).run(suite)
    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1
    if retval != 0:
        raise RuntimeError('One or more tests failed.')
