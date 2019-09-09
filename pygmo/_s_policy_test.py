# -*- coding: utf-8 -*-

# Copyright 2017-2018 PaGMO development team
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


class _s_pol(object):

    def select(self, inds, nx, nix, nobj, nec, nic, tol):
        return inds


class s_policy_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.s_policy` class.

    """

    def runTest(self):
        self.run_basic_tests()
        self.run_extract_tests()
        self.run_name_info_tests()
        self.run_pickle_tests()

    def run_basic_tests(self):
        # Tests for minimal s_policy, and mandatory methods.
        import numpy as np
        from .core import s_policy, select_best
        # Def construction.
        r = s_policy()
        self.assertTrue(r.extract(select_best) is not None)
        self.assertTrue(r.extract(int) is None)

        # First a few non-s_pols.
        self.assertRaises(NotImplementedError, lambda: s_policy(1))
        self.assertRaises(NotImplementedError, lambda: s_policy([]))
        self.assertRaises(TypeError, lambda: s_policy(int))
        # Some policies missing methods, wrong arity, etc.

        class nr0(object):
            pass
        self.assertRaises(NotImplementedError, lambda: s_policy(nr0()))

        class nr1(object):

            select = 45
        self.assertRaises(NotImplementedError, lambda: s_policy(nr1()))

        # The minimal good citizen.
        glob = []

        class r(object):

            def __init__(self, g):
                self.g = g

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                self.g.append(2)
                return inds

        r_inst = r(glob)
        s_pol = s_policy(r_inst)

        # Test the keyword arg.
        s_pol = s_policy(udsp=select_best())
        s_pol = s_policy(udsp=r_inst)

        # Check a few s_pol properties.
        self.assertEqual(s_pol.get_extra_info(), "")
        self.assertTrue(s_pol.extract(int) is None)
        self.assertTrue(s_pol.extract(select_best) is None)
        self.assertFalse(s_pol.extract(r) is None)
        self.assertTrue(s_pol.is_(r))

        # Check the select method.
        self.assertTrue(isinstance(s_pol.select(
            ([], [], []), 1, 0, 1, 0, 0, []), tuple))
        r_out = s_pol.select(([1, 2], [[.1, .2], [.3, .4]], [
                              [1.1], [2.2]]), 2, 0, 1, 0, 0, [])
        self.assertTrue(np.all(r_out[0] == np.array([1, 2])))
        self.assertTrue(r_out[1].dtype == np.dtype(float))
        self.assertTrue(r_out[2].dtype == np.dtype(float))
        self.assertTrue(np.all(r_out[1] == np.array([[.1, .2], [.3, .4]])))
        self.assertTrue(np.all(r_out[2] == np.array([[1.1], [2.2]])))
        self.assertTrue(len(r_out) == 3)
        # Assert that r_inst was deep-copied into s_pol:
        # the instance in s_pol will have its own copy of glob
        # and it will not be a reference the outside object.
        self.assertEqual(len(glob), 0)
        self.assertEqual(len(s_pol.extract(r).g), 2)
        self.assertEqual(s_pol.extract(r).g, [2]*2)

        s_pol = s_policy(select_best())
        self.assertTrue(s_pol.get_extra_info() != "")
        self.assertTrue(s_pol.extract(int) is None)
        self.assertTrue(s_pol.extract(r) is None)
        self.assertFalse(s_pol.extract(select_best) is None)
        self.assertTrue(s_pol.is_(select_best))

        # Wrong retvals for select().

        class r(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return []
        s_pol = s_policy(r())
        self.assertRaises(RuntimeError, lambda: s_pol.select(inds=([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), nx=2, nix=0, nobj=1, nec=0, nic=0, tol=[]))
        # Try also flipping around the named argument.
        self.assertRaises(RuntimeError, lambda: s_pol.select(nx=2, inds=([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), nec=0, nix=0, nobj=1, nic=0, tol=[]))

        class r(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return [1]
        s_pol = s_policy(r())
        self.assertRaises(RuntimeError, lambda: s_pol.select(([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), 2, 0, 1, 0, 0, []))

        class r(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return [1, 2]
        s_pol = s_policy(r())
        self.assertRaises(RuntimeError, lambda: s_pol.select(([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), 2, 0, 1, 0, 0, []))

        class r(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return [1, 2, 3, 4]
        s_pol = s_policy(r())
        self.assertRaises(RuntimeError, lambda: s_pol.select(([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), 2, 0, 1, 0, 0, []))

        class r(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return 1
        s_pol = s_policy(r())
        self.assertRaises(RuntimeError, lambda: s_pol.select(([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), 2, 0, 1, 0, 0, []))

        class r(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return ([1], [], [])
        s_pol = s_policy(r())
        with self.assertRaises(ValueError) as cm:
            s_pol.select(([1, 2], [[.1], [.3, .4]], [
                [1.1], [2.2]]), 2, 0, 1, 0, 0, [])
        err = cm.exception
        self.assertTrue(
            "not all the individuals passed to a selection policy of type " in str(err))
        with self.assertRaises(ValueError) as cm:
            s_pol.select(([1, 2], [[.1, .2]], [
                [1.1], [2.2]]), 2, 0, 1, 0, 0, [])
        err = cm.exception
        self.assertTrue(
            "must all have the same sizes, but instead their sizes are " in str(err))
        with self.assertRaises(ValueError) as cm:
            s_pol.select(([1, 2], [[.1, .2], [.3, .4]], [
                [1.1], [2.2]]), 2, 0, 1, 0, 0, [])
        err = cm.exception
        self.assertTrue(
            "must all have the same sizes, but instead their sizes are " in str(err))

        # Test wrong array construction of IDs.
        class r(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return (np.array([1], dtype=int), [[1, 2]], [[1]])
        s_pol = s_policy(r())
        with self.assertRaises(RuntimeError) as cm:
            s_pol.select(([1, 2], [[.1, .2], [.3, .4]], [
                [1.1], [2.2]]), 2, 0, 1, 0, 0, [])

        # Test wrong array construction of IDs.
        class r(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return (np.array([1], dtype='float64'), [[1, 2]], [[1]])
        s_pol = s_policy(r())
        with self.assertRaises(RuntimeError) as cm:
            s_pol.select(([1, 2], [[.1, .2], [.3, .4]], [
                [1.1], [2.2]]), 2, 0, 1, 0, 0, [])

        # Test construction of array ID from list.
        class r(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return ([1], [[1, 2]], [[1]])
        s_pol = s_policy(r())
        ret = s_pol.select(([1, 2], [[.1, .2], [.3, .4]], [
                           [1.1], [2.2]]), 2, 0, 1, 0, 0, [])
        self.assertEqual(ret[0][0], 1)

        # Test construction of array ID from array.
        class r(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return (np.array([1], dtype='ulonglong'), [[1, 2]], [[1]])
        s_pol = s_policy(r())
        ret = s_pol.select(([1, 2], [[.1, .2], [.3, .4]], [
                           [1.1], [2.2]]), 2, 0, 1, 0, 0, [])
        self.assertEqual(ret[0][0], 1)

        # Test that construction from another pygmo.s_policy fails.
        with self.assertRaises(TypeError) as cm:
            s_policy(s_pol)
        err = cm.exception
        self.assertTrue(
            "a pygmo.s_policy cannot be used as a UDSP for another pygmo.s_policy (if you need to copy a selection policy please use the standard Python copy()/deepcopy() functions)" in str(err))

    def run_extract_tests(self):
        from .core import s_policy, _test_s_policy, select_best
        import sys

        # First we try with a C++ test s_pol.
        t = s_policy(_test_s_policy())
        # Verify the refcount of p is increased after extract().
        rc = sys.getrefcount(t)
        ts_pol = t.extract(_test_s_policy)
        self.assertEqual(sys.getrefcount(t), rc + 1)
        del ts_pol
        self.assertEqual(sys.getrefcount(t), rc)
        # Verify we are modifying the inner object.
        t.extract(_test_s_policy).set_n(5)
        self.assertEqual(t.extract(_test_s_policy).get_n(), 5)

        class ts_policy(object):

            def __init__(self):
                self._n = 1

            def get_n(self):
                return self._n

            def set_n(self, n):
                self._n = n

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return inds

        # Test with Python s_policy.
        t = s_policy(ts_policy())
        rc = sys.getrefcount(t)
        ts_pol = t.extract(ts_policy)
        # Reference count does not increase because
        # ts_policy is stored as a proper Python object
        # with its own refcount.
        self.assertTrue(sys.getrefcount(t) == rc)
        self.assertTrue(ts_pol.get_n() == 1)
        ts_pol.set_n(12)
        self.assert_(t.extract(ts_policy).get_n() == 12)

        # Check that we can extract Python UDTs also via Python's object type.
        t = s_policy(ts_policy())
        self.assertTrue(not t.extract(object) is None)
        # Check we are referring to the same object.
        self.assertEqual(id(t.extract(object)), id(t.extract(ts_policy)))
        # Check that it will not work with exposed C++ selection policies.
        t = s_policy(select_best())
        self.assertTrue(t.extract(object) is None)
        self.assertTrue(not t.extract(select_best) is None)

    def run_name_info_tests(self):
        from .core import s_policy

        class t(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return inds

        s_pol = s_policy(t())
        self.assertTrue(s_pol.get_name() != '')
        self.assertTrue(s_pol.get_extra_info() == '')

        class t(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return inds

            def get_name(self):
                return 'pippo'

        s_pol = s_policy(t())
        self.assertTrue(s_pol.get_name() == 'pippo')
        self.assertTrue(s_pol.get_extra_info() == '')

        class t(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return inds

            def get_extra_info(self):
                return 'pluto'

        s_pol = s_policy(t())
        self.assertTrue(s_pol.get_name() != '')
        self.assertTrue(s_pol.get_extra_info() == 'pluto')

        class t(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return inds

            def get_name(self):
                return 'pippo'

            def get_extra_info(self):
                return 'pluto'

        s_pol = s_policy(t())
        self.assertTrue(s_pol.get_name() == 'pippo')
        self.assertTrue(s_pol.get_extra_info() == 'pluto')

    def run_pickle_tests(self):
        from .core import s_policy, select_best
        from pickle import dumps, loads
        t_ = s_policy(select_best())
        t = loads(dumps(t_))
        self.assertEqual(repr(t), repr(t_))
        self.assertTrue(t.is_(select_best))

        t_ = s_policy(_s_pol())
        t = loads(dumps(t_))
        self.assertEqual(repr(t), repr(t_))
        self.assertTrue(t.is_(_s_pol))
