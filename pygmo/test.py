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


class _quick_prob:

    def fitness(self, dv):
        return [sum(dv)]

    def get_bounds(self):
        return ([0] * 10, [1] * 10)


class _raise_exception:
    counter = 0

    def __init__(self, throw_at=3000):
        self.throw_at = throw_at

    def fitness(self, dv):
        if type(self).counter == self.throw_at:
            raise
        type(self).counter += 1
        return [0]

    def get_bounds(self):
        return ([0], [1])


class _raise_exception_2:
    counter = 0

    def fitness(self, dv):
        if _raise_exception_2.counter == 300:
            raise
        _raise_exception_2.counter += 1
        return [0]

    def get_bounds(self):
        return ([0], [1])


class core_test_case(_ut.TestCase):
    """Test case for core PyGMO functionality.

    """

    def runTest(self):
        import sys
        from numpy import random, all, array
        from .core import _builtin, _test_to_vd, _type, _str, _callable, _deepcopy, _test_object_serialization as tos
        from . import __version__
        self.assertTrue(__version__ != "")
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
        self.run_pickle_test()

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

        def prob_setter():
            pop.problem = problem(zdt(param=10))
        self.assertRaises(AttributeError, prob_setter)

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
        # Test bogus x, f dimensions.
        pop = population(rosenbrock(5), size=5)
        self.assertRaises(ValueError, lambda: pop.push_back([]))
        self.assertRaises(ValueError, lambda: pop.push_back([], []))
        self.assertRaises(ValueError, lambda: pop.push_back([1] * 5, []))
        self.assertRaises(ValueError, lambda: pop.push_back([1] * 5, [1, 2]))

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

    def run_pickle_test(self):
        from .core import population, rosenbrock, translate
        from pickle import dumps, loads
        pop = population(rosenbrock(), size=12, seed=42)
        p = loads(dumps(pop))
        self.assertEqual(repr(pop), repr(p))
        pop = population(translate(rosenbrock(2), 2 * [.1]), size=12, seed=42)
        p = loads(dumps(pop))
        self.assertEqual(repr(pop), repr(p))
        pop = population(_prob(), size=12, seed=42)
        p = loads(dumps(pop))
        self.assertEqual(repr(pop), repr(p))
        pop = population(translate(_prob(), 2 * [.1]), size=12, seed=42)
        p = loads(dumps(pop))
        self.assertEqual(repr(pop), repr(p))


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


class bee_colony_test_case(_ut.TestCase):
    """Test case for the UDA bee_colony

    """

    def runTest(self):
        from .core import bee_colony
        uda = bee_colony()
        uda = bee_colony(gen=1, limit=10)
        uda = bee_colony(gen=1, limit=10, seed=32)
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


class sga_test_case(_ut.TestCase):
    """Test case for the UDA sga

    """

    def runTest(self):
        from .core import sga
        uda = sga()
        uda = sga(gen=1, cr=.90, eta_c=1., m=0.02, param_m=1., param_s=2, crossover="exponential",
                  mutation="polynomial", selection="tournament")
        uda = sga(gen=1, cr=.90, eta_c=1., m=0.02, param_m=1., param_s=2, crossover="exponential",
                  mutation="polynomial", selection="tournament", seed=32)
        self.assertEqual(uda.get_seed(), 32)
        seed = uda.get_seed()


class nsga2_test_case(_ut.TestCase):
    """Test case for the UDA nsga2

    """

    def runTest(self):
        from .core import nsga2
        uda = nsga2()
        uda = nsga2(gen=1, cr=0.95, eta_c=10, m=0.01, eta_m=10)
        uda = nsga2(gen=1, cr=0.95, eta_c=10, m=0.01,
                    eta_m=10, int_dim=0, seed=32)
        self.assertEqual(uda.get_seed(), 32)
        seed = uda.get_seed()


class nlopt_test_case(_ut.TestCase):
    """Test case for the UDA nlopt

    """

    def runTest(self):
        from .core import nlopt, algorithm, luksan_vlcek1, problem, population
        n = nlopt()
        self.assertEqual(n.get_solver_name(), "cobyla")
        n = nlopt(solver="slsqp")
        self.assertEqual(n.get_solver_name(), "slsqp")
        self.assertRaises(ValueError, lambda: nlopt("dsadsa"))

        self.assertEqual(n.get_last_opt_result(), 1)

        self.assertEqual(n.ftol_abs, 0.)
        n.ftol_abs = 1E-6
        self.assertEqual(n.ftol_abs, 1E-6)

        def _():
            n.ftol_abs = float('nan')
        self.assertRaises(ValueError, _)

        self.assertEqual(n.ftol_rel, 0.)
        n.ftol_rel = 1E-6
        self.assertEqual(n.ftol_rel, 1E-6)

        def _():
            n.ftol_rel = float('nan')
        self.assertRaises(ValueError, _)

        self.assertEqual(n.maxeval, 0)
        n.maxeval = 42
        self.assertEqual(n.maxeval, 42)

        self.assertEqual(n.maxtime, 0)
        n.maxtime = 43
        self.assertEqual(n.maxtime, 43)

        self.assertEqual(n.replacement, "best")
        n.replacement = "worst"
        self.assertEqual(n.replacement, "worst")

        def _():
            n.replacement = "rr"
        self.assertRaises(ValueError, _)
        n.replacement = 12
        self.assertEqual(n.replacement, 12)

        def _():
            n.replacement = -1
        self.assertRaises(OverflowError, _)

        self.assertEqual(n.selection, "best")
        n.selection = "worst"
        self.assertEqual(n.selection, "worst")

        def _():
            n.selection = "rr"
        self.assertRaises(ValueError, _)
        n.selection = 12
        self.assertEqual(n.selection, 12)

        def _():
            n.selection = -1
        self.assertRaises(OverflowError, _)

        n.set_random_sr_seed(12)
        self.assertRaises(OverflowError, lambda: n.set_random_sr_seed(-1))

        self.assertEqual(n.stopval, -float('inf'))
        n.stopval = 1E-6
        self.assertEqual(n.stopval, 1E-6)

        def _():
            n.stopval = float('nan')
        self.assertRaises(ValueError, _)

        self.assertEqual(n.xtol_abs, 0.)
        n.xtol_abs = 1E-6
        self.assertEqual(n.xtol_abs, 1E-6)

        def _():
            n.xtol_abs = float('nan')
        self.assertRaises(ValueError, _)

        self.assertEqual(n.xtol_rel, 1E-8)
        n.xtol_rel = 1E-6
        self.assertEqual(n.xtol_rel, 1E-6)

        def _():
            n.xtol_rel = float('nan')
        self.assertRaises(ValueError, _)

        n = nlopt("slsqp")
        algo = algorithm(n)
        algo.set_verbosity(5)
        prob = problem(luksan_vlcek1(20))
        prob.c_tol = [1E-6] * 18
        pop = population(prob, 20)
        pop = algo.evolve(pop)
        self.assertTrue(len(algo.extract(nlopt).get_log()) != 0)

        # Pickling.
        from pickle import dumps, loads
        algo = algorithm(nlopt("slsqp"))
        algo.set_verbosity(5)
        prob = problem(luksan_vlcek1(20))
        prob.c_tol = [1E-6] * 18
        pop = population(prob, 20)
        algo.evolve(pop)
        self.assertEqual(str(algo), str(loads(dumps(algo))))
        self.assertEqual(algo.extract(nlopt).get_log(), loads(
            dumps(algo)).extract(nlopt).get_log())

        # Local optimizer.
        self.assertTrue(nlopt("slsqp").local_optimizer is None)
        self.assertTrue(nlopt("auglag").local_optimizer is None)
        n = nlopt("auglag")
        loc = nlopt("slsqp")
        n.local_optimizer = loc
        self.assertFalse(n.local_optimizer is None)
        self.assertEqual(str(algorithm(loc)), str(
            algorithm(n.local_optimizer)))
        pop = population(prob, 20, seed=4)
        algo = algorithm(n)
        algo.evolve(pop)
        self.assertTrue(algo.extract(nlopt).get_last_opt_result() >= 0)
        n = nlopt("auglag_eq")
        loc = nlopt("slsqp")
        n.local_optimizer = loc
        self.assertFalse(n.local_optimizer is None)
        self.assertEqual(str(algorithm(loc)), str(
            algorithm(n.local_optimizer)))
        pop = population(prob, 20, seed=4)
        algo = algorithm(n)
        algo.evolve(pop)
        self.assertTrue(algo.extract(nlopt).get_last_opt_result() >= 0)

        # Refcount.
        import sys
        nl = nlopt("auglag")
        loc = nlopt("slsqp")
        nl.local_optimizer = loc
        old_rc = sys.getrefcount(nl)
        foo = nl.local_optimizer
        self.assertEqual(old_rc + 1, sys.getrefcount(nl))
        del nl
        self.assertTrue(len(str(foo)) != 0)
        del foo


class ipopt_test_case(_ut.TestCase):
    """Test case for the UDA ipopt

    """

    def runTest(self):
        from .core import ipopt, algorithm, luksan_vlcek1, problem, population
        ip = ipopt()
        # Check the def-cted state.
        self.assertEqual(ip.get_last_opt_result(), 0)
        self.assertEqual(ip.get_log(), [])
        self.assertEqual(ip.get_numeric_options(), {})
        self.assertEqual(ip.get_integer_options(), {})
        self.assertEqual(ip.get_numeric_options(), {})
        self.assertEqual(ip.selection, "best")
        self.assertEqual(ip.replacement, "best")
        self.assertTrue(len(str(algorithm(ip))) != 0)

        # Options testing.
        ip.set_string_option("marge", "simpson")
        self.assertEqual(ip.get_string_options(), {"marge": "simpson"})
        ip.set_string_options({"homer": "simpson", "bart": "simpson"})
        self.assertEqual(ip.get_string_options(), {
                         "marge": "simpson", "bart": "simpson", "homer": "simpson"})
        ip.reset_string_options()
        self.assertEqual(ip.get_string_options(), {})

        ip.set_integer_option("marge", 0)
        self.assertEqual(ip.get_integer_options(), {"marge": 0})
        ip.set_integer_options({"homer": 1, "bart": 2})
        self.assertEqual(ip.get_integer_options(), {
                         "marge": 0, "bart": 2, "homer": 1})
        ip.reset_integer_options()
        self.assertEqual(ip.get_integer_options(), {})

        ip.set_numeric_option("marge", 0.)
        self.assertEqual(ip.get_numeric_options(), {"marge": 0.})
        ip.set_numeric_options({"homer": 1., "bart": 2.})
        self.assertEqual(ip.get_numeric_options(), {
                         "marge": 0., "bart": 2., "homer": 1.})
        ip.reset_numeric_options()
        self.assertEqual(ip.get_numeric_options(), {})

        # Select/replace.
        self.assertEqual(ip.replacement, "best")
        ip.replacement = "worst"
        self.assertEqual(ip.replacement, "worst")

        def _():
            ip.replacement = "rr"
        self.assertRaises(ValueError, _)
        ip.replacement = 12
        self.assertEqual(ip.replacement, 12)

        def _():
            ip.replacement = -1
        self.assertRaises(OverflowError, _)

        self.assertEqual(ip.selection, "best")
        ip.selection = "worst"
        self.assertEqual(ip.selection, "worst")

        def _():
            ip.selection = "rr"
        self.assertRaises(ValueError, _)
        ip.selection = 12
        self.assertEqual(ip.selection, 12)

        def _():
            ip.selection = -1
        self.assertRaises(OverflowError, _)

        ip.set_random_sr_seed(12)
        self.assertRaises(OverflowError, lambda: ip.set_random_sr_seed(-1))

        ip = ipopt()
        algo = algorithm(ip)
        algo.set_verbosity(5)
        prob = problem(luksan_vlcek1(20))
        prob.c_tol = [1E-6] * 18
        pop = population(prob, 20)
        pop = algo.evolve(pop)
        self.assertTrue(len(algo.extract(ipopt).get_log()) != 0)

        # Pickling.
        from pickle import dumps, loads
        ip = ipopt()
        ip.set_numeric_option("tol", 1E-7)
        algo = algorithm(ip)
        algo.set_verbosity(5)
        prob = problem(luksan_vlcek1(20))
        prob.c_tol = [1E-6] * 18
        pop = population(prob, 20)
        algo.evolve(pop)
        self.assertEqual(str(algo), str(loads(dumps(algo))))
        self.assertEqual(algo.extract(ipopt).get_log(), loads(
            dumps(algo)).extract(ipopt).get_log())
        self.assertEqual(algo.extract(ipopt).get_numeric_options(), loads(
            dumps(algo)).extract(ipopt).get_numeric_options())


class null_problem_test_case(_ut.TestCase):
    """Test case for the null problem

    """

    def runTest(self):
        from .core import null_problem as np, problem
        n = np()
        n = np(1)
        n = np(nobj=2)
        n = np(nobj=2, nec=2)
        n = np(nobj=2, nec=2, nic=2)
        self.assertRaises(ValueError, lambda: np(0))
        self.assertTrue(problem(np()).get_nobj() == 1)
        self.assertTrue(problem(np(23)).get_nobj() == 23)


class estimate_sparsity_test_case(_ut.TestCase):
    """Test case for the hypervolume utilities

    """

    def runTest(self):
        import pygmo as pg
        import numpy as np

        def my_fun(x):
            return [x[0] + x[3], x[2], x[1]]
        res = pg.estimate_sparsity(
            callable=my_fun, x=[0.1, 0.1, 0.1, 0.1], dx=1e-8)
        self.assertTrue(
            (res == np.array([[0, 0], [0, 3], [1, 2], [2, 1]])).all())


class estimate_gradient_test_case(_ut.TestCase):
    """Test case for the hypervolume utilities

    """

    def runTest(self):
        import pygmo as pg
        import numpy as np

        def my_fun(x):
            return [x[0] + x[3], x[2], x[1]]
        out = pg.estimate_gradient(callable=my_fun, x=[0] * 4, dx=1e-8)
        res = np.array([1.,  0.,  0.,  1.,  0.,  0.,
                        1.,  0.,  0.,  1.,  0.,  0.])
        self.assertTrue((abs(out - res) < 1e-8).all())
        out = pg.estimate_gradient_h(callable=my_fun, x=[0] * 4, dx=1e-8)
        self.assertTrue((abs(out - res) < 1e-8).all())


class mo_utils_test_case(_ut.TestCase):
    """Test case for the multi-objective utilities (only the interface is tested)

    """

    def runTest(self):
        from .core import fast_non_dominated_sorting, pareto_dominance, non_dominated_front_2d, crowding_distance, sort_population_mo, select_best_N_mo, decompose_objectives, decomposition_weights, nadir, ideal, population, dtlz
        ndf, dl, dc, ndr = fast_non_dominated_sorting(
            points=[[0, 1], [-1, 3], [2.3, -0.2], [1.1, -0.12], [1.1, 2.12], [-1.1, -1.1]])
        self.assertTrue(pareto_dominance(obj1=[1, 2], obj2=[2, 2]))
        non_dominated_front_2d(
            points=[[0, 5], [1, 4], [2, 3], [3, 2], [4, 1], [2, 2]])
        crowding_distance(points=[[0, 5], [1, 4], [2, 3], [3, 2], [4, 1]])
        pop = population(prob=dtlz(prob_id=3, dim=10, fdim=4), size=20)
        sort_population_mo(points=pop.get_f())
        select_best_N_mo(points=pop.get_f(), N=13)
        decompose_objectives(objs=[1, 2, 3], weights=[0.1, 0.1, 0.8], ref_point=[
                             5, 5, 5], method="weighted")
        decomposition_weights(n_f=2, n_w=6, method="low discrepancy", seed=33)
        nadir(points=[[1, 1], [-1, 1], [2.2, 3], [0.1, -0.1]])
        ideal(points=[[1, 1], [-1, 1], [2.2, 3], [0.1, -0.1]])


class con_utils_test_case(_ut.TestCase):
    """Test case for the constrained utilities (only the interface is tested)

    """

    def runTest(self):
        from .core import compare_fc, sort_population_con
        compare_fc(f1=[1, 1, 1], f2=[1, 2.1, -1.2], nec=1, tol=[0] * 2)
        sort_population_con(
            input_f=[[1.2, 0.1, -1], [0.2, 1.1, 1.1], [2, -0.5, -2]], nec=1, tol=[1e-8] * 2)


class global_rng_test_case(_ut.TestCase):
    """Test case for the global random number generator

    """

    def runTest(self):
        from .core import set_global_rng_seed, population, ackley
        set_global_rng_seed(seed=32)
        pop = population(prob=ackley(5), size=20)
        f1 = pop.champion_f
        set_global_rng_seed(seed=32)
        pop = population(prob=ackley(5), size=20)
        f2 = pop.champion_f
        self.assertTrue(f1 == f2)


class hypervolume_test_case(_ut.TestCase):
    """Test case for the hypervolume utilities

    """

    def runTest(self):
        from .core import hypervolume, hv2d, hv3d, hvwfg, bf_fpras, bf_approx
        from .core import population, zdt
        import numpy as np
        pop = population(prob=zdt(prob_id=1, param=10), size=20)
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
        udp = dtlz(prob_id=3, dim=9, fdim=3, alpha=5)
        udp.p_distance([0.2] * 9)
        udp.p_distance(population(udp, 20))


class minlp_rastrigin_test_case(_ut.TestCase):
    """Test case for the MINLP Rastrigin

    """

    def runTest(self):
        from .core import minlp_rastrigin, problem, population
        udp = minlp_rastrigin(dim_c=2, dim_i=3)
        prob = problem(udp)
        self.assertTrue(prob.get_nx() == 5)
        self.assertTrue(prob.get_nix() == 3)
        self.assertTrue(prob.get_ncx() == 2)
        pop = population(udp, 1)
        self.assertTrue(int(pop.get_x()[0][-1]) == pop.get_x()[0][-1])
        self.assertTrue(int(pop.get_x()[0][-2]) == pop.get_x()[0][-2])
        self.assertTrue(int(pop.get_x()[0][-3]) == pop.get_x()[0][-3])
        self.assertTrue(int(pop.get_x()[0][0]) != pop.get_x()[0][0])
        self.assertTrue(int(pop.get_x()[0][1]) != pop.get_x()[0][1])


class random_decision_vector_test_case(_ut.TestCase):
    """Test case for random_decision_vector

    """

    def runTest(self):
        from .core import random_decision_vector, set_global_rng_seed
        set_global_rng_seed(42)
        x = random_decision_vector(lb = [1.1,2.1,-3], ub = [2.1, 3.4,5], nix = 1)
        self.assertTrue(int(x[-1]) == x[-1])
        self.assertTrue(int(x[1]) != x[1])
        set_global_rng_seed(42)
        y = random_decision_vector(lb = [1.1,2.1,-3], ub = [2.1, 3.4,5], nix = 1)
        self.assertTrue((x == y).all())
        nan = float("nan")
        inf = float("inf")
        self.assertRaises(ValueError, lambda : random_decision_vector([1, 2], [0, 3]))
        self.assertRaises(ValueError, lambda : random_decision_vector([1, -inf], [0, 32]))
        self.assertRaises(ValueError, lambda : random_decision_vector([1, 2, 3], [0, 3]))
        self.assertRaises(ValueError, lambda : random_decision_vector([0, 2, 3], [1, 4, nan]))
        self.assertRaises(ValueError, lambda : random_decision_vector([0, 2, nan], [1, 4, 4]))
        self.assertRaises(ValueError, lambda : random_decision_vector([0, nan, 3], [1, nan, 4]))
        self.assertRaises(ValueError, lambda : random_decision_vector([0, 2, 3], [1, 4, 5], 4))
        self.assertRaises(ValueError, lambda : random_decision_vector([0, 2, 3.1], [1, 4, 5], 1))
        self.assertRaises(ValueError, lambda : random_decision_vector([0, 2, 3], [1, 4, 5.2], 1))
        self.assertRaises(ValueError, lambda : random_decision_vector([0, -1.1, 3], [1, 2, 5], 2))
        self.assertRaises(ValueError, lambda : random_decision_vector([0, -1.1, -inf], [1, 2, inf], 2))
        self.assertRaises(ValueError, lambda : random_decision_vector([0, -1.1, inf], [1, 2, inf], 2))


class luksan_vlcek1_test_case(_ut.TestCase):
    """Test case for the UDP Luksan Vlcek 1

    """

    def runTest(self):
        from .core import luksan_vlcek1, population
        udp = luksan_vlcek1(dim=3)


class cec2006_test_case(_ut.TestCase):
    """Test case for the UDP cec2006

    """

    def runTest(self):
        from .core import cec2006, population
        udp = cec2006(prob_id=3)
        best = udp.best_known


class cec2009_test_case(_ut.TestCase):
    """Test case for the UDP cec2009

    """

    def runTest(self):
        from .core import cec2009, population
        udp = cec2009(prob_id=3, is_constrained=True, dim=15)


class cec2013_test_case(_ut.TestCase):
    """Test case for the UDP cec2013

    """

    def runTest(self):
        try:
            # NOTE: cec2013 is not always present (see MSVC issue).
            from .core import cec2013, population
        except ImportError:
            return
        udp = cec2013(prob_id=3, dim=10)


class translate_test_case(_ut.TestCase):
    """Test case for the translate meta-problem

    """

    def runTest(self):
        from .core import problem, rosenbrock, translate, null_problem, decompose
        from numpy import array

        t = translate()
        self.assertFalse(t.inner_problem.extract(null_problem) is None)
        self.assertTrue(all(t.translation == array([0.])))
        t = translate(prob=rosenbrock(), translation=[1, 2])
        self.assertFalse(t.inner_problem.extract(rosenbrock) is None)
        self.assertTrue(all(t.translation == array([1., 2.])))
        t = translate(rosenbrock(), [1, 2])
        self.assertTrue(problem(t).is_(translate))
        self.assertFalse(problem(t).extract(translate) is None)
        self.assertTrue(t.inner_problem.is_(rosenbrock))
        self.assertFalse(t.inner_problem.extract(rosenbrock) is None)
        self.assertTrue(all(t.translation == array([1., 2.])))
        t = translate(translation=[1, 2], prob=rosenbrock())
        self.assertFalse(t.inner_problem.extract(rosenbrock) is None)
        self.assertTrue(all(t.translation == array([1., 2.])))

        # Nested translation.
        t = translate(translate(rosenbrock(), [1, 2]), [1, 2])
        self.assertTrue(t.inner_problem.is_(translate))
        self.assertFalse(t.inner_problem.extract(translate) is None)
        self.assertFalse(t.inner_problem.extract(
            translate).inner_problem.extract(rosenbrock) is None)

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        t = translate(p(), [-1, -1])
        self.assertFalse(t.inner_problem.extract(p) is None)
        self.assertTrue(all(t.translation == array([-1., -1.])))
        t = translate(translation=[-1, -1], prob=p())
        self.assertTrue(t.inner_problem.is_(p))
        self.assertFalse(t.inner_problem.extract(p) is None)
        self.assertTrue(all(t.translation == array([-1., -1.])))

        # Verify construction from problem is allowed.
        translate(problem(null_problem()), [0.])

        # Verify translation of decompose.
        t = translate(decompose(null_problem(2), [0.2, 0.8], [0., 0.]), [0.])
        self.assertTrue(t.inner_problem.is_(decompose))
        self.assertFalse(t.inner_problem.extract(decompose) is None)
        self.assertFalse(t.inner_problem.extract(
            decompose).inner_problem.extract(null_problem) is None)


class unconstrain_test_case(_ut.TestCase):
    """Test case for the unconstrain meta-problem

    """

    def runTest(self):
        from .core import hock_schittkowsky_71, unconstrain, null_problem, problem, translate
        from numpy import array

        d = unconstrain()
        self.assertFalse(d.inner_problem.extract(null_problem) is None)
        d = unconstrain(prob=hock_schittkowsky_71(),
                        method="weighted", weights=[1., 1.])
        self.assertTrue(problem(d).is_(unconstrain))
        self.assertFalse(problem(d).extract(unconstrain) is None)
        self.assertTrue(d.inner_problem.is_(hock_schittkowsky_71))
        self.assertFalse(d.inner_problem.extract(hock_schittkowsky_71) is None)

        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42, 43]

            def get_nobj(self):
                return 2

            def get_nic(self):
                return 2

            def get_nec(self):
                return 2

        d = unconstrain(p(), "kuri")
        self.assertTrue(d.inner_problem.is_(p))
        self.assertFalse(d.inner_problem.extract(p) is None)

        # Verify construction from problem is allowed.
        unconstrain(problem(null_problem(2, 2, 2)), "kuri")

        # Verify chaining of metas
        t = unconstrain(translate(null_problem(
            2, 2, 2), [0.]), "death penalty")
        self.assertTrue(t.inner_problem.is_(translate))
        self.assertFalse(t.inner_problem.extract(translate) is None)
        self.assertFalse(t.inner_problem.extract(
            translate).inner_problem.extract(null_problem) is None)


class decompose_test_case(_ut.TestCase):
    """Test case for the decompose meta-problem

    """

    def runTest(self):
        from .core import zdt, decompose, null_problem, problem, translate
        from numpy import array

        d = decompose()
        self.assertFalse(d.inner_problem.extract(null_problem) is None)
        self.assertTrue(all(d.z == array([0., 0.])))
        d = decompose(zdt(1, 2), [0.5, 0.5], [0.1, 0.1], "weighted", False)
        self.assertTrue(problem(d).is_(decompose))
        self.assertFalse(problem(d).extract(decompose) is None)
        self.assertTrue(d.inner_problem.is_(zdt))
        self.assertFalse(d.inner_problem.extract(zdt) is None)
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
        self.assertTrue(d.inner_problem.is_(p))
        self.assertFalse(d.inner_problem.extract(p) is None)
        self.assertTrue(all(d.z == array([0.1, 0.1])))
        self.assertTrue(all(d.original_fitness([1., 1.]) == array([42, 43])))

        # Verify construction from problem is allowed.
        decompose(problem(null_problem(2)), [0.2, 0.8], [0., 0.])

        # Verify decomposition of translate.
        t = decompose(translate(null_problem(2), [0.]), [0.2, 0.8], [0., 0.])
        self.assertTrue(t.inner_problem.is_(translate))
        self.assertFalse(t.inner_problem.extract(translate) is None)
        self.assertFalse(t.inner_problem.extract(
            translate).inner_problem.extract(null_problem) is None)


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
        self.assertFalse(a.inner_algorithm.extract(compass_search) is None)
        self.assertTrue(a.inner_algorithm.is_(compass_search))
        self.assertTrue(a.inner_algorithm.extract(de) is None)
        self.assertFalse(a.inner_algorithm.is_(de))
        self.assertEqual(a.get_log(), [])
        self.assertTrue(all(a.get_perturb() == array([0.01])))
        seed = a.get_seed()
        self.assertEqual(a.get_verbosity(), 0)
        a.set_perturb([.2])
        self.assertTrue(all(a.get_perturb() == array([0.2])))
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.basic)
        self.assertTrue(al.extract(mbh).inner_algorithm.extract(
            compass_search) is not None)
        self.assertTrue(al.extract(mbh).inner_algorithm.extract(de) is None)
        self.assertTrue(str(seed) in str(al))
        al.set_verbosity(4)
        self.assertEqual(al.extract(mbh).get_verbosity(), 4)

        # From C++ algo.
        seed = 123321
        a = mbh(algo=de(), stop=5, perturb=.4)
        a = mbh(stop=5, perturb=(.4, .2), algo=de())
        a = mbh(algo=de(), stop=5, seed=seed, perturb=(.4, .2))
        self.assertTrue(a.inner_algorithm.extract(compass_search) is None)
        self.assertFalse(a.inner_algorithm.is_(compass_search))
        self.assertFalse(a.inner_algorithm.extract(de) is None)
        self.assertTrue(a.inner_algorithm.is_(de))
        self.assertEqual(a.get_log(), [])
        self.assertTrue(all(a.get_perturb() == array([.4, .2])))
        self.assertEqual(a.get_seed(), seed)
        self.assertEqual(a.get_verbosity(), 0)
        a.set_perturb([.2])
        self.assertTrue(all(a.get_perturb() == array([0.2])))
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.basic)
        self.assertTrue(al.extract(mbh).inner_algorithm.extract(
            compass_search) is None)
        self.assertTrue(al.extract(
            mbh).inner_algorithm.extract(de) is not None)
        self.assertTrue(str(seed) in str(al))
        al.set_verbosity(4)
        self.assertEqual(al.extract(mbh).get_verbosity(), 4)

        # From Python algo.
        class algo(object):

            def evolve(self, pop):
                return pop

        seed = 123321
        a = mbh(algo=algo(), stop=5, perturb=.4)
        a = mbh(stop=5, perturb=(.4, .2), algo=algo())
        a = mbh(algo=algo(), stop=5, seed=seed, perturb=(.4, .2))
        self.assertTrue(a.inner_algorithm.extract(compass_search) is None)
        self.assertFalse(a.inner_algorithm.is_(compass_search))
        self.assertFalse(a.inner_algorithm.extract(algo) is None)
        self.assertTrue(a.inner_algorithm.is_(algo))
        self.assertEqual(a.get_log(), [])
        self.assertTrue(all(a.get_perturb() == array([.4, .2])))
        self.assertEqual(a.get_seed(), seed)
        self.assertEqual(a.get_verbosity(), 0)
        a.set_perturb([.2])
        self.assertTrue(all(a.get_perturb() == array([0.2])))
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.none)
        self.assertTrue(al.extract(mbh).inner_algorithm.extract(
            compass_search) is None)
        self.assertTrue(al.extract(
            mbh).inner_algorithm.extract(algo) is not None)
        self.assertTrue(str(seed) in str(al))
        al.set_verbosity(4)
        self.assertEqual(al.extract(mbh).get_verbosity(), 4)

        # Construction from algorithm is allowed.
        mbh(algorithm(null_algorithm()), stop=5, perturb=.4)


class cstrs_self_adaptive_test_case(_ut.TestCase):
    """Test case for the cstrs_self_adaptive meta-algorithm

    """

    def runTest(self):
        from . import cstrs_self_adaptive, de, compass_search, algorithm, thread_safety as ts, null_algorithm
        from numpy import array

        class algo(object):

            def evolve(pop):
                return pop

        # Def ctor.
        a = cstrs_self_adaptive()
        self.assertFalse(a.inner_algorithm.extract(de) is None)
        self.assertTrue(a.inner_algorithm.is_(de))
        self.assertTrue(a.inner_algorithm.extract(compass_search) is None)
        self.assertFalse(a.inner_algorithm.is_(compass_search))
        self.assertEqual(a.get_log(), [])
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.basic)
        self.assertTrue(al.extract(cstrs_self_adaptive).inner_algorithm.extract(
            de) is not None)
        self.assertTrue(al.extract(cstrs_self_adaptive).inner_algorithm.extract(
            compass_search) is None)
        al.set_verbosity(4)

        # From C++ algo.
        seed = 123321
        a = cstrs_self_adaptive(algo=de())
        a = cstrs_self_adaptive(algo=de(), iters=1500)
        a = cstrs_self_adaptive(seed=32, algo=de(), iters=12)
        self.assertTrue(a.inner_algorithm.extract(compass_search) is None)
        self.assertFalse(a.inner_algorithm.is_(compass_search))
        self.assertFalse(a.inner_algorithm.extract(de) is None)
        self.assertTrue(a.inner_algorithm.is_(de))
        self.assertEqual(a.get_log(), [])
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.basic)
        self.assertTrue(al.extract(cstrs_self_adaptive).inner_algorithm.extract(
            compass_search) is None)
        self.assertTrue(al.extract(
            cstrs_self_adaptive).inner_algorithm.extract(de) is not None)
        al.set_verbosity(4)

        # From Python algo.
        class algo(object):

            def evolve(self, pop):
                return pop

        seed = 123321
        a = cstrs_self_adaptive(algo=de())
        a = cstrs_self_adaptive(algo=de(), iters=1500)
        a = cstrs_self_adaptive(seed=32, algo=de(), iters=12)
        self.assertTrue(a.inner_algorithm.extract(compass_search) is None)
        self.assertFalse(a.inner_algorithm.is_(compass_search))
        self.assertFalse(a.inner_algorithm.extract(de) is None)
        self.assertTrue(a.inner_algorithm.is_(de))
        self.assertEqual(a.get_log(), [])
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.basic)
        self.assertTrue(al.extract(cstrs_self_adaptive).inner_algorithm.extract(
            compass_search) is None)
        self.assertTrue(al.extract(
            cstrs_self_adaptive).inner_algorithm.extract(de) is not None)
        al.set_verbosity(4)

        # Construction from algorithm is allowed.
        cstrs_self_adaptive(algo=algorithm(null_algorithm()), seed=5, iters=4)


class archipelago_test_case(_ut.TestCase):
    """Test case for the archipelago class.

    """

    def __init__(self, level):
        _ut.TestCase.__init__(self)
        self._level = level

    def runTest(self):
        self.run_init_tests()
        self.run_evolve_tests()
        self.run_access_tests()
        self.run_push_back_tests()
        self.run_io_tests()
        self.run_pickle_tests()
        self.run_champions_tests()
        self.run_status_tests()
        if self._level > 0:
            self.run_torture_test_0()
            self.run_torture_test_1()

    def run_init_tests(self):
        from . import archipelago, de, rosenbrock, population, null_problem, thread_island, mp_island
        a = archipelago()
        self.assertEqual(len(a), 0)
        self.assertRaises(IndexError, lambda: a[0])
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        self.assertEqual(len(a), 5)
        self.assertTrue(a[0].get_algorithm().is_(de))
        self.assertTrue(a[0].get_population().problem.is_(rosenbrock))
        self.assertEqual(len(a[0].get_population()), 10)
        a = archipelago(5, pop=population(), algo=de())
        self.assertEqual(len(a), 5)
        self.assertTrue(a[0].get_algorithm().is_(de))
        self.assertTrue(a[0].get_population().problem.is_(null_problem))
        self.assertEqual(len(a[0].get_population()), 0)
        a = archipelago(5, algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=thread_island(), seed=5)
        self.assertEqual(len(a), 5)
        self.assertTrue(a[0].get_algorithm().is_(de))
        self.assertTrue(a[0].get_population().problem.is_(rosenbrock))
        self.assertEqual(len(a[0].get_population()), 10)
        # Check unique seeds.
        seeds = list([_.get_population().get_seed() for _ in a])
        self.assertEqual(len(seeds), len(set(seeds)))
        # Check seeding is deterministic.
        a2 = archipelago(5, algo=de(), prob=rosenbrock(),
                         pop_size=10, seed=5)
        seeds2 = list([_.get_population().get_seed() for _ in a2])
        self.assertEqual(seeds2, seeds)
        self.assertTrue(all([(t[0].get_population().get_x() == t[
                        1].get_population().get_x()).all() for t in zip(a, a2)]))
        self.assertTrue(all([(t[0].get_population().get_f() == t[
                        1].get_population().get_f()).all() for t in zip(a, a2)]))
        self.assertTrue(all([(t[0].get_population().get_ID() == t[
                        1].get_population().get_ID()).all() for t in zip(a, a2)]))
        # Check the 'size' keyword is not accepted.
        self.assertRaises(KeyError, lambda: archipelago(5, algo=de(), prob=rosenbrock(),
                                                        size=10, udi=thread_island(), seed=5))
        # Check without seed argument, seeding is non-deterministic.
        a = archipelago(5, algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=thread_island())
        a2 = archipelago(5, algo=de(), prob=rosenbrock(),
                         pop_size=10, udi=thread_island())
        seeds = sorted(list([_.get_population().get_seed() for _ in a]))
        seeds2 = sorted(list([_.get_population().get_seed() for _ in a2]))
        self.assertTrue(all([t[0] != t[1] for t in zip(seeds, seeds2)]))
        self.assertTrue(all([(t[0].get_population().get_x() != t[
                        1].get_population().get_x()).all() for t in zip(a, a2)]))
        self.assertTrue(all([(t[0].get_population().get_f() != t[
                        1].get_population().get_f()).all() for t in zip(a, a2)]))
        self.assertTrue(all([(t[0].get_population().get_ID() != t[
                        1].get_population().get_ID()).all() for t in zip(a, a2)]))
        import sys
        import os
        # The mp island requires either Windows or at least Python 3.4.
        if os.name != 'nt' and (sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 4)):
            return
        a = archipelago(5, algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=mp_island(), seed=5)
        self.assertEqual(len(a), 5)
        self.assertTrue(a[0].get_algorithm().is_(de))
        self.assertTrue(a[0].get_population().problem.is_(rosenbrock))
        self.assertEqual(len(a[0].get_population()), 10)
        seeds = list([_.get_population().get_seed() for _ in a])
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertRaises(KeyError, lambda: archipelago(
            5, pop=population(), algo=de(), seed=1))

    def run_evolve_tests(self):
        from . import archipelago, de, rosenbrock, mp_island, evolve_status
        from copy import deepcopy
        a = archipelago()
        self.assertTrue(a.status == evolve_status.idle)
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait()
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait_check()
        # Copy while evolving.
        a.evolve(10)
        a.evolve(10)
        a2 = deepcopy(a)
        a.wait_check()
        import sys
        import os
        # The mp island requires either Windows or at least Python 3.4.
        if os.name != 'nt' and (sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 4)):
            return
        a = archipelago(5, udi=mp_island(), algo=de(),
                        prob=rosenbrock(), pop_size=10)
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait()
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait_check()
        # Copy while evolving.
        a.evolve(10)
        a.evolve(10)
        a2 = deepcopy(a)
        a.wait_check()
        # Throws on wait_check().
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=3)
        a.evolve()
        self.assertRaises(ValueError, lambda: a.wait_check())

    def run_access_tests(self):
        from . import archipelago, de, rosenbrock
        import sys
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        i0, i1, i2 = a[0], a[1], a[2]
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        for isl in (i0, i1, i2):
            self.assertTrue(isl.get_algorithm().is_(de))
            self.assertTrue(isl.get_population().problem.is_(rosenbrock))
            self.assertEqual(len(isl.get_population()), 10)
        # Check refcount when returning internal ref.
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        old_rc = sys.getrefcount(a)
        i0, i1, i2, i3 = a[0], a[1], a[2], a[3]
        self.assertEqual(sys.getrefcount(a) - 4, old_rc)
        del a
        self.assertTrue(str(i0) != "")
        self.assertTrue(str(i1) != "")
        self.assertTrue(str(i2) != "")
        self.assertTrue(str(i3) != "")
        del i0, i1, i2, i3

    def run_push_back_tests(self):
        from . import archipelago, de, rosenbrock
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        # Push back while evolving.
        a.evolve(10)
        a.evolve(10)
        a.evolve(10)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.wait_check()
        self.assertEqual(len(a), 18)
        for i in range(5):
            self.assertTrue(a[i].get_algorithm().is_(de))
            self.assertTrue(a[i].get_population().problem.is_(rosenbrock))
            self.assertEqual(len(a[i].get_population()), 10)
        for i in range(5, 18):
            self.assertTrue(a[i].get_algorithm().is_(de))
            self.assertTrue(a[i].get_population().problem.is_(rosenbrock))
            self.assertEqual(len(a[i].get_population()), 11)

    def run_io_tests(self):
        from . import archipelago, de, rosenbrock
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        self.assertFalse(repr(a) == "")

    def run_pickle_tests(self):
        from . import archipelago, de, rosenbrock, mp_island
        from pickle import dumps, loads
        import sys
        import os
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        self.assertEqual(repr(a), repr(loads(dumps(a))))
        # The mp island requires either Windows or at least Python 3.4.
        if os.name != 'nt' and (sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 4)):
            return
        a = archipelago(5, algo=de(), prob=_prob(),
                        pop_size=10, udi=mp_island())
        self.assertEqual(repr(a), repr(loads(dumps(a))))

    def run_champions_tests(self):
        from . import archipelago, de, rosenbrock, zdt
        from numpy import ndarray
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        cf = a.get_champions_f()
        self.assertEqual(type(cf), list)
        self.assertEqual(len(cf), 5)
        self.assertEqual(type(cf[0]), ndarray)
        cx = a.get_champions_x()
        self.assertEqual(type(cx), list)
        self.assertEqual(len(cx), 5)
        self.assertEqual(type(cx[0]), ndarray)
        a.push_back(algo=de(), prob=rosenbrock(10), size=20)
        cx = a.get_champions_x()
        self.assertEqual(len(cx[4]), 2)
        self.assertEqual(len(cx[5]), 10)
        a.push_back(algo=de(), prob=zdt(), size=20)
        self.assertRaises(ValueError, lambda: a.get_champions_x())
        self.assertRaises(ValueError, lambda: a.get_champions_f())

    def run_status_tests(self):
        from . import archipelago, de, rosenbrock, evolve_status
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=3)
        self.assertTrue(a.status == evolve_status.idle)
        a.evolve()
        a.wait()
        self.assertTrue(a.status == evolve_status.idle_error)
        self.assertRaises(ValueError, lambda: a.wait_check())
        self.assertTrue(a.status == evolve_status.idle)

    def run_torture_test_0(self):
        from . import archipelago, de, ackley

        # pure C++
        archi = archipelago(n=1000, algo=de(
            10), prob=ackley(5), pop_size=10, seed=32)
        archi.evolve()
        archi.wait_check()

        # python prob
        archi2 = archipelago(n=1000, algo=de(
            10), prob=_quick_prob(), pop_size=10, seed=32)
        archi2.evolve()
        archi2.wait_check()

        # python prob with exceptions (will throw in osx as too many threads
        # will be opened)
        def _():
            archi3 = archipelago(n=1000, algo=simulated_annealing(
                10, 1, 50), prob=_raise_exception(throw_at=1001), pop_size=1, seed=32)
            archi3.evolve()
            archi3.wait_check()

        self.assertRaises(BaseException, _)

    def run_torture_test_1(self):
        # A torture test inspired by the heisenbug detected by Dario on OSX.

        from . import archipelago, sade, ackley

        archi = archipelago(n=5, algo=sade(
            50), prob=_raise_exception_2(), pop_size=20)
        archi.evolve()
        self.assertRaises(BaseException, lambda: archi.wait_check())

        archi = archipelago(n=5, algo=sade(
            50), prob=_raise_exception_2(), pop_size=20)
        archi.evolve()
        self.assertRaises(BaseException, lambda: archi.wait_check())
        archi.wait_check()

        archi = archipelago(n=1100, algo=sade(
            500), prob=ackley(50), pop_size=50)
        archi = archipelago(n=5, algo=sade(
            50), prob=_raise_exception_2(), pop_size=20)
        archi.evolve()
        archi = archipelago(n=1100, algo=sade(
            500), prob=ackley(50), pop_size=50)
        archi.evolve()


def run_test_suite(level=0):
    """Run the full test suite.

    This function will raise an exception if at least one test fails.

    Args:
        level(``int``): the test level (higher values run longer tests)

    """
    from . import _problem_test, _algorithm_test, _island_test, set_global_rng_seed

    # Make test runs deterministic.
    # NOTE: we'll need to place the async/migration tests at the end, so that at
    # least the first N tests are really deterministic.
    set_global_rng_seed(42)

    retval = 0
    suite = _ut.TestLoader().loadTestsFromTestCase(core_test_case)
    suite.addTest(_problem_test.problem_test_case())
    suite.addTest(_algorithm_test.algorithm_test_case())
    suite.addTest(_island_test.island_test_case())
    suite.addTest(_island_test.mp_island_test_case(level))
    suite.addTest(_island_test.ipyparallel_island_test_case(level))
    suite.addTest(pso_test_case())
    suite.addTest(bee_colony_test_case())
    suite.addTest(compass_search_test_case())
    suite.addTest(sa_test_case())
    suite.addTest(moead_test_case())
    suite.addTest(sga_test_case())
    suite.addTest(population_test_case())
    suite.addTest(archipelago_test_case(level))
    suite.addTest(null_problem_test_case())
    suite.addTest(hypervolume_test_case())
    suite.addTest(mo_utils_test_case())
    suite.addTest(con_utils_test_case())
    suite.addTest(global_rng_test_case())
    suite.addTest(estimate_sparsity_test_case())
    suite.addTest(estimate_gradient_test_case())
    suite.addTest(random_decision_vector_test_case())
    try:
        from .core import cmaes
        suite.addTest(cmaes_test_case())
    except ImportError:
        pass
    suite.addTest(dtlz_test_case())
    suite.addTest(cec2006_test_case())
    suite.addTest(cec2009_test_case())
    suite.addTest(cec2013_test_case())
    suite.addTest(luksan_vlcek1_test_case())
    suite.addTest(minlp_rastrigin_test_case())
    suite.addTest(translate_test_case())
    suite.addTest(decompose_test_case())
    suite.addTest(unconstrain_test_case())
    suite.addTest(mbh_test_case())
    suite.addTest(cstrs_self_adaptive_test_case())
    try:
        from .core import nlopt
        suite.addTest(nlopt_test_case())
    except ImportError:
        pass
    try:
        from .core import ipopt
        suite.addTest(ipopt_test_case())
    except ImportError:
        pass
    test_result = _ut.TextTestRunner(verbosity=2).run(suite)

    # Re-seed to random just in case anyone ever uses this function
    # in an interactive session or something.
    import random
    set_global_rng_seed(random.randint(0, 2**30))

    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1
    if retval != 0:
        raise RuntimeError('One or more tests failed.')
