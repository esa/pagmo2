# -*- coding: utf-8 -*-

# Copyright 2017-2020 PaGMO development team
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


class _bf(object):
    def __call__(self, p, dvs):
        dim = p.get_nx()
        nf = p.get_nf()
        ndvs = len(dvs) // dim

        return ([1] * nf) * ndvs


class bfe_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.bfe` class.

    """

    def runTest(self):
        self.run_basic_tests()
        self.run_extract_tests()
        self.run_name_info_tests()
        self.run_thread_safety_tests()
        self.run_pickle_test()

    def run_basic_tests(self):
        # Tests for minimal bfe, and mandatory methods.
        from .core import bfe, default_bfe, thread_bfe, rosenbrock, problem
        # Def construction.
        b = bfe()
        self.assertTrue(b.extract(default_bfe) is not None)
        self.assertTrue(b.extract(thread_bfe) is None)

        # First a few non-bfes.
        self.assertRaises(NotImplementedError, lambda: bfe(1))
        self.assertRaises(NotImplementedError, lambda: bfe("hello world"))
        self.assertRaises(NotImplementedError, lambda: bfe([]))
        self.assertRaises(TypeError, lambda: bfe(int))

        class nb0(object):
            pass

        self.assertRaises(NotImplementedError, lambda: bfe(nb0()))

        # The minimal good citizen.
        glob = []

        class b(object):

            def __init__(self, g):
                self.g = g

            def __call__(self, p, dvs):
                self.g.append(1)
                dim = p.get_nx()
                nf = p.get_nf()
                ndvs = len(dvs) // dim

                return ([1] * nf) * ndvs

        b_inst = b(glob)
        bf = bfe(b_inst)
        # Test the keyword arg.
        bf = bfe(udbfe=b(glob))
        bf = bfe(udbfe=b_inst)

        self.assertTrue(bf.is_(b))
        self.assertTrue(not bf.is_(int))
        self.assertTrue(id(bf.extract(b)) != id(b_inst))
        self.assertTrue(bf.extract(int) is None)

        # Call operator.
        self.assertTrue(all(bf(prob=rosenbrock(5), dvs=[5]*35) == [1]*7))
        # A few more times.
        self.assertTrue(all(bf(rosenbrock(5), [5]*35) == [1]*7))
        self.assertTrue(all(bf(problem(rosenbrock(5)), [5]*35) == [1]*7))

        # Assert that b_inst was deep-copied into prob:
        # the instance in bf will have its own copy of glob
        # and it will not be a reference the outside object.
        self.assertEqual(len(glob), 0)
        self.assertEqual(len(bf.extract(b).g), 3)

        # Various testing bits for the call operator.

        class b(object):

            def __call__(self, p, dvs):
                import numpy as np

                dim = p.get_nx()
                nf = p.get_nf()
                ndvs = len(dvs) // dim

                # Return as numpy array.
                return np.array(([1] * nf) * ndvs)

        self.assertTrue(all(bfe(b())(prob=rosenbrock(5), dvs=[5]*35) == [1]*7))

        class b(object):

            def __call__(self, p, dvs):
                dim = p.get_nx()
                nf = p.get_nf()
                ndvs = len(dvs) // dim

                # Return as tuple.
                return tuple(([1] * nf) * ndvs)

        self.assertTrue(all(bfe(b())(prob=rosenbrock(5), dvs=[5]*35) == [1]*7))

        # Try with a function.
        def b_func(p, dvs):
            dim = p.get_nx()
            nf = p.get_nf()
            ndvs = len(dvs) // dim
            return ([1] * nf) * ndvs

        self.assertTrue(
            all(bfe(b_func)(prob=rosenbrock(5), dvs=[5]*35) == [1]*7))
        self.assertTrue(bfe(b_func).is_(type(b_func)))

        # A few failure modes.
        def b_func(p, dvs):
            dim = p.get_nx()
            nf = p.get_nf()
            ndvs = len(dvs) // dim
            # Wrong size of the returned list.
            return ([1] * nf) * (ndvs + 1)

        self.assertRaises(ValueError, lambda: bfe(
            b_func)(prob=rosenbrock(5), dvs=[5]*35))

        def b_func(p, dvs):
            dim = p.get_nx()
            nf = p.get_nf()
            ndvs = len(dvs) // dim
            # Return non-iterable object.
            return 42

        self.assertRaises(AttributeError, lambda: bfe(
            b_func)(prob=rosenbrock(5), dvs=[5]*35))

        # Test that construction from another pygmo.bfe fails.
        with self.assertRaises(TypeError) as cm:
            bfe(bfe(b_func))
        err = cm.exception
        self.assertTrue(
            "a pygmo.bfe cannot be used as a UDBFE for another pygmo.bfe (if you need to copy a bfe please use the standard Python copy()/deepcopy() functions)" in str(err))

    def run_extract_tests(self):
        from .core import bfe, _test_bfe, thread_bfe
        import sys

        # First we try with a C++ test bfe.
        bf = bfe(_test_bfe())
        # Verify the refcount of bf is increased after extract().
        rc = sys.getrefcount(bf)
        tbf = bf.extract(_test_bfe)
        self.assertTrue(sys.getrefcount(bf) == rc + 1)
        del tbf
        self.assertTrue(sys.getrefcount(bf) == rc)
        # Verify we are modifying the inner object.
        bf.extract(_test_bfe).set_n(5)
        self.assertTrue(bf.extract(_test_bfe).get_n() == 5)

        class tb(object):
            def __init__(self):
                self._n = 1

            def get_n(self):
                return self._n

            def set_n(self, n):
                self._n = n

            def __call__(self, p, dvs):
                dim = p.get_nx()
                nf = p.get_nf()
                ndvs = len(dvs) // dim

                return ([1] * nf) * ndvs

        # Test with Python bfe.
        bf = bfe(tb())
        rc = sys.getrefcount(bf)
        tbf = bf.extract(tb)
        # Reference count does not increase because
        # tb is stored as a proper Python object
        # with its own refcount.
        self.assertTrue(sys.getrefcount(bf) == rc)
        self.assertTrue(tbf.get_n() == 1)
        tbf.set_n(12)
        self.assertTrue(bf.extract(tb).get_n() == 12)

        # Check that we can extract Python UDPs also via Python's object type.
        bf = bfe(tb())
        self.assertTrue(not bf.extract(object) is None)
        # Check we are referring to the same object.
        self.assertEqual(id(bf.extract(object)), id(bf.extract(tb)))
        # Check that it will not work with exposed C++ bfes.
        bf = bfe(thread_bfe())
        self.assertTrue(bf.extract(object) is None)
        self.assertTrue(not bf.extract(thread_bfe) is None)

    def run_name_info_tests(self):
        from .core import bfe

        class tb(object):
            def __call__(self, p, dvs):
                dim = p.get_nx()
                nf = p.get_nf()
                ndvs = len(dvs) // dim

                return ([1] * nf) * ndvs

        bf = bfe(tb())
        self.assert_(bf.get_name() != '')
        self.assert_(bf.get_extra_info() == '')

        class tb(object):
            def __call__(self, p, dvs):
                dim = p.get_nx()
                nf = p.get_nf()
                ndvs = len(dvs) // dim

                return ([1] * nf) * ndvs

            def get_name(self):
                return 'pippo'

        bf = bfe(tb())
        self.assertTrue(bf.get_name() == 'pippo')
        self.assertTrue(bf.get_extra_info() == '')

        class tb(object):
            def __call__(self, p, dvs):
                dim = p.get_nx()
                nf = p.get_nf()
                ndvs = len(dvs) // dim

                return ([1] * nf) * ndvs

            def get_extra_info(self):
                return 'pluto'

        bf = bfe(tb())
        self.assertTrue(bf.get_name() != '')
        self.assertTrue(bf.get_extra_info() == 'pluto')

        class tb(object):
            def __call__(self, p, dvs):
                dim = p.get_nx()
                nf = p.get_nf()
                ndvs = len(dvs) // dim

                return ([1] * nf) * ndvs

            def get_name(self):
                return 'pippo'

            def get_extra_info(self):
                return 'pluto'

        bf = bfe(tb())
        self.assertTrue(bf.get_name() == 'pippo')
        self.assertTrue(bf.get_extra_info() == 'pluto')

    def run_thread_safety_tests(self):
        from .core import bfe, thread_bfe, _tu_test_bfe
        from . import thread_safety as ts

        class tb(object):
            def __call__(self, p, dvs):
                dim = p.get_nx()
                nf = p.get_nf()
                ndvs = len(dvs) // dim

                return ([1] * nf) * ndvs

        self.assertTrue(bfe(tb()).get_thread_safety() == ts.none)
        self.assertTrue(
            bfe(thread_bfe()).get_thread_safety() == ts.basic)
        self.assertTrue(
            bfe(_tu_test_bfe()).get_thread_safety() == ts.none)

    def run_pickle_test(self):
        from .core import bfe, thread_bfe
        from pickle import dumps, loads
        bf = bfe(thread_bfe())
        bf = loads(dumps(bf))
        self.assertEqual(repr(bf), repr(bfe(thread_bfe())))
        self.assertTrue(bf.is_(thread_bfe))

        bf = bfe(_bf())
        bf = loads(dumps(bf))
        self.assertEqual(repr(bf), repr(bfe(_bf())))
        self.assertTrue(bf.is_(_bf))


class thread_bfe_test_case(_ut.TestCase):
    """Test case for the thread_bfe UDBFE

    """

    def runTest(self):
        from .core import thread_bfe, bfe, member_bfe, rosenbrock, batch_random_decision_vector, problem
        udbfe = thread_bfe()
        b = bfe(udbfe=udbfe)
        self.assertTrue(b.is_(thread_bfe))
        self.assertFalse(b.is_(member_bfe))

        prob = problem(rosenbrock(5))
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)
        fvs.shape = (6, )
        dvs.shape = (6, 5)
        for dv, fv in zip(dvs, fvs):
            self.assertTrue(fv == prob.fitness(dv))

        self.assertTrue(
            b.get_name() == "Multi-threaded batch fitness evaluator")
        self.assertTrue(b.get_extra_info() == "")

        class p(object):
            def fitness(self, x):
                return [0]

            def get_bounds(self):
                return ([0], [1])

            def get_name(self):
                return 'pippo'

        prob = problem(p())

        with self.assertRaises(ValueError) as cm:
            b(prob, [0])
        err = cm.exception
        self.assertTrue(
            "Cannot use a thread_bfe on the problem 'pippo', which does not provide the required level of thread safety" in str(err))


class member_bfe_test_case(_ut.TestCase):
    """Test case for the member_bfe UDBFE

    """

    def runTest(self):
        from .core import thread_bfe, bfe, member_bfe, rosenbrock, batch_random_decision_vector, problem
        udbfe = member_bfe()
        b = bfe(udbfe=udbfe)
        self.assertFalse(b.is_(thread_bfe))
        self.assertTrue(b.is_(member_bfe))

        prob = problem(rosenbrock(5))
        dvs = batch_random_decision_vector(prob, 6)

        with self.assertRaises(NotImplementedError) as cm:
            b(prob, dvs)
        err = cm.exception
        self.assertTrue(
            "The batch_fitness() method has been invoked, but it is not implemented in a UDP of type 'Multidimensional Rosenbrock Function'" in str(err))

        class p(object):
            def fitness(self, x):
                return [0]

            def get_bounds(self):
                return ([0], [1])

            def batch_fitness(self, dvs):
                return [42] * len(dvs)

        prob = problem(p())
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)
        for f in fvs:
            self.assertTrue(f == 42)


class mp_bfe_test_case(_ut.TestCase):
    """Test case for the mp_bfe UDBFE

    """

    def runTest(self):
        from .core import thread_bfe, bfe, rosenbrock, batch_random_decision_vector, problem
        from . import mp_bfe

        mp_bfe.shutdown_pool()
        mp_bfe.shutdown_pool()
        mp_bfe.shutdown_pool()

        udbfe = mp_bfe()
        b = bfe(udbfe=udbfe)
        self.assertFalse(b.is_(thread_bfe))
        self.assertTrue(b.is_(mp_bfe))

        mp_bfe.init_pool()
        mp_bfe.init_pool()
        mp_bfe.init_pool()

        prob = problem(rosenbrock(5))
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)
        fvs.shape = (6, )
        dvs.shape = (6, 5)
        for dv, fv in zip(dvs, fvs):
            self.assertTrue(fv == prob.fitness(dv))
        self.assertEqual(prob.get_fevals(), 6)

        class p(object):
            def fitness(self, x):
                return [0]

            def get_bounds(self):
                return ([0], [1])

        mp_bfe.resize_pool(2)
        self.assertEqual(mp_bfe.get_pool_size(), 2)
        mp_bfe.resize_pool(2)
        self.assertEqual(mp_bfe.get_pool_size(), 2)
        mp_bfe.shutdown_pool()
        mp_bfe.resize_pool(16)
        mp_bfe.shutdown_pool()

        prob = problem(p())
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)

        for fv in fvs:
            self.assertTrue(fv == 0.)

        # Try different chunksize as well.
        udbfe = mp_bfe(chunksize=2)
        b = bfe(udbfe=udbfe)
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)

        for fv in fvs:
            self.assertTrue(fv == 0.)

        udbfe = mp_bfe(chunksize=None)
        b = bfe(udbfe=udbfe)
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)

        for fv in fvs:
            self.assertTrue(fv == 0.)

        # Error handling.
        with self.assertRaises(TypeError) as cm:
            b = bfe(udbfe=mp_bfe(chunksize='pippo'))
        err = cm.exception
        self.assertTrue(
            "The 'chunksize' argument must be None or an int, but it is of type '{}' instead".format(str) in str(err))

        with self.assertRaises(ValueError) as cm:
            b = bfe(udbfe=mp_bfe(chunksize=0))
        err = cm.exception
        self.assertTrue(
            "The 'chunksize' parameter must be a positive integer, but its value is 0 instead" in str(err))

        with self.assertRaises(ValueError) as cm:
            b = bfe(udbfe=mp_bfe(chunksize=-1))
        err = cm.exception
        self.assertTrue(
            "The 'chunksize' parameter must be a positive integer, but its value is -1 instead" in str(err))

        self.assertEqual(
            b.get_name(), "Multiprocessing batch fitness evaluator")
        self.assertTrue(
            "Number of processes in the pool" in b.get_extra_info())

        # Test exception transport in the pool.
        class _bfe_throw_prob(object):
            def fitness(self, x):
                raise ValueError("oh snap")

            def get_bounds(self):
                return ([0], [1])

        prob = problem(_bfe_throw_prob())
        dvs = batch_random_decision_vector(prob, 6)
        with self.assertRaises(ValueError) as cm:
            b(prob, dvs)
        err = cm.exception
        self.assertTrue(
            "oh snap" in str(err))


class ipyparallel_bfe_test_case(_ut.TestCase):
    """Test case for the ipyparallel_bfe UDBFE

    """

    def runTest(self):
        try:
            import ipyparallel
        except ImportError:
            return

        from .core import thread_bfe, bfe, rosenbrock, batch_random_decision_vector, problem
        from . import ipyparallel_bfe

        ipyparallel_bfe.shutdown_view()
        ipyparallel_bfe.shutdown_view()
        ipyparallel_bfe.shutdown_view()

        to = .5
        try:
            # Try with kwargs for the client.
            ipyparallel_bfe.init_view(client_kwargs={'timeout': to})
        except OSError:
            return

        udbfe = ipyparallel_bfe()
        b = bfe(udbfe=udbfe)
        self.assertFalse(b.is_(thread_bfe))
        self.assertTrue(b.is_(ipyparallel_bfe))

        prob = problem(rosenbrock(5))
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)
        fvs.shape = (6, )
        dvs.shape = (6, 5)
        for dv, fv in zip(dvs, fvs):
            self.assertTrue(fv == prob.fitness(dv))
        self.assertEqual(prob.get_fevals(), 6)

        class p(object):
            def fitness(self, x):
                return [0]

            def get_bounds(self):
                return ([0], [1])

        ipyparallel_bfe.init_view()

        prob = problem(p())
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)

        for fv in fvs:
            self.assertTrue(fv == 0.)

        ipyparallel_bfe.shutdown_view()
        ipyparallel_bfe.shutdown_view()

        prob = problem(p())
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)

        for fv in fvs:
            self.assertTrue(fv == 0.)

        self.assertEqual(b.get_name(), "Ipyparallel batch fitness evaluator")
        self.assertTrue(b.get_extra_info() != '')

        # Test exception transport.
        class _bfe_throw_prob(object):
            def fitness(self, x):
                raise ValueError("oh snap")

            def get_bounds(self):
                return ([0], [1])

        prob = problem(_bfe_throw_prob())
        dvs = batch_random_decision_vector(prob, 6)
        with self.assertRaises(ipyparallel.error.CompositeError) as cm:
            b(prob, dvs)
        err = cm.exception
        self.assertTrue(
            "oh snap" in str(err))


class default_bfe_test_case(_ut.TestCase):
    """Test case for the default_bfe UDBFE

    """

    def runTest(self):
        from .core import thread_bfe, bfe, default_bfe, rosenbrock, batch_random_decision_vector, problem

        udbfe = default_bfe()
        b = bfe(udbfe=udbfe)
        self.assertFalse(b.is_(thread_bfe))
        self.assertTrue(b.is_(default_bfe))

        prob = problem(rosenbrock(5))
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)
        fvs.shape = (6, )
        dvs.shape = (6, 5)
        for dv, fv in zip(dvs, fvs):
            self.assertTrue(fv == prob.fitness(dv))

        class p(object):
            def fitness(self, x):
                return [0]

            def get_bounds(self):
                return ([0], [1])

        prob = problem(p())
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)

        for fv in fvs:
            self.assertTrue(fv == 0.)

        class p(object):
            def fitness(self, x):
                return [0]

            def get_bounds(self):
                return ([0], [1])

            def batch_fitness(self, dvs):
                return [42] * len(dvs)

        prob = problem(p())
        dvs = batch_random_decision_vector(prob, 6)
        fvs = b(prob, dvs)
        for f in fvs:
            self.assertTrue(f == 42)
