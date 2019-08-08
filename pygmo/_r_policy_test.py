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


class _r_pol(object):

    def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
        return inds


class r_policy_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.r_policy` class.

    """

    def runTest(self):
        self.run_basic_tests()
        self.run_extract_tests()
        self.run_name_info_tests()
        self.run_pickle_tests()

    def run_basic_tests(self):
        # Tests for minimal r_policy, and mandatory methods.
        import numpy as np
        from .core import r_policy, fair_replace
        # Def construction.
        r = r_policy()
        self.assertTrue(r.extract(fair_replace) is not None)
        self.assertTrue(r.extract(int) is None)

        # First a few non-r_pols.
        self.assertRaises(NotImplementedError, lambda: r_policy(1))
        self.assertRaises(NotImplementedError, lambda: r_policy([]))
        self.assertRaises(TypeError, lambda: r_policy(int))
        # Some policies missing methods, wrong arity, etc.

        class nr0(object):
            pass
        self.assertRaises(NotImplementedError, lambda: r_policy(nr0()))

        class nr1(object):

            replace = 45
        self.assertRaises(NotImplementedError, lambda: r_policy(nr1()))

        # The minimal good citizen.
        glob = []

        class r(object):

            def __init__(self, g):
                self.g = g

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                self.g.append(2)
                return inds

        r_inst = r(glob)
        r_pol = r_policy(r_inst)

        # Test the keyword arg.
        r_pol = r_policy(udrp=fair_replace())
        r_pol = r_policy(udrp=r_inst)

        # Check a few r_pol properties.
        self.assertEqual(r_pol.get_extra_info(), "")
        self.assertTrue(r_pol.extract(int) is None)
        self.assertTrue(r_pol.extract(fair_replace) is None)
        self.assertFalse(r_pol.extract(r) is None)
        self.assertTrue(r_pol.is_(r))

        # Check the replace method.
        self.assertTrue(isinstance(r_pol.replace(
            ([], [], []), 1, 0, 1, 0, 0, [], ([], [], [])), tuple))
        r_out = r_pol.replace(([1, 2], [[.1, .2], [.3, .4]], [
                              [1.1], [2.2]]), 2, 0, 1, 0, 0, [], ([], [], []))
        self.assertTrue(np.all(r_out[0] == np.array([1, 2])))
        self.assertTrue(r_out[1].dtype == np.dtype(float))
        self.assertTrue(r_out[2].dtype == np.dtype(float))
        self.assertTrue(np.all(r_out[1] == np.array([[.1, .2], [.3, .4]])))
        self.assertTrue(np.all(r_out[2] == np.array([[1.1], [2.2]])))
        self.assertTrue(len(r_out) == 3)
        # Assert that r_inst was deep-copied into r_pol:
        # the instance in r_pol will have its own copy of glob
        # and it will not be a reference the outside object.
        self.assertEqual(len(glob), 0)
        self.assertEqual(len(r_pol.extract(r).g), 2)
        self.assertEqual(r_pol.extract(r).g, [2]*2)

        r_pol = r_policy(fair_replace())
        self.assertTrue(r_pol.get_extra_info() != "")
        self.assertTrue(r_pol.extract(int) is None)
        self.assertTrue(r_pol.extract(r) is None)
        self.assertFalse(r_pol.extract(fair_replace) is None)
        self.assertTrue(r_pol.is_(fair_replace))

        # Wrong retvals for replace().

        class r(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return []
        r_pol = r_policy(r())
        self.assertRaises(RuntimeError, lambda: r_pol.replace(inds=([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), nx=2, nix=0, nobj=1, nec=0, nic=0, tol=[], mig=([], [], [])))
        # Try also flipping around the named argument.
        self.assertRaises(RuntimeError, lambda: r_pol.replace(nx=2, inds=([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), nec=0, nix=0, nobj=1, nic=0, mig=([], [], []), tol=[]))

        class r(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return [1]
        r_pol = r_policy(r())
        self.assertRaises(RuntimeError, lambda: r_pol.replace(([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), 2, 0, 1, 0, 0, [], ([], [], [])))

        class r(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return [1, 2]
        r_pol = r_policy(r())
        self.assertRaises(RuntimeError, lambda: r_pol.replace(([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), 2, 0, 1, 0, 0, [], ([], [], [])))

        class r(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return [1, 2, 3, 4]
        r_pol = r_policy(r())
        self.assertRaises(RuntimeError, lambda: r_pol.replace(([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), 2, 0, 1, 0, 0, [], ([], [], [])))

        class r(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return 1
        r_pol = r_policy(r())
        self.assertRaises(RuntimeError, lambda: r_pol.replace(([1, 2], [[.1, .2], [.3, .4]], [
            [1.1], [2.2]]), 2, 0, 1, 0, 0, [], ([], [], [])))

        class r(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return ([1], [], [])
        r_pol = r_policy(r())
        with self.assertRaises(ValueError) as cm:
            r_pol.replace(([1, 2], [[.1], [.3, .4]], [
                [1.1], [2.2]]), 2, 0, 1, 0, 0, [], ([], [], []))
        err = cm.exception
        self.assertTrue(
            "not all the individuals passed to a replacement policy of type " in str(err))
        with self.assertRaises(ValueError) as cm:
            r_pol.replace(([1, 2], [[.1, .2]], [
                [1.1], [2.2]]), 2, 0, 1, 0, 0, [], ([], [], []))
        err = cm.exception
        self.assertTrue(
            "must all have the same sizes, but instead their sizes are " in str(err))
        with self.assertRaises(ValueError) as cm:
            r_pol.replace(([1, 2], [[.1, .2], [.3, .4]], [
                [1.1], [2.2]]), 2, 0, 1, 0, 0, [], ([], [], []))
        err = cm.exception
        self.assertTrue(
            "must all have the same sizes, but instead their sizes are " in str(err))

        # Test wrong array construction of IDs.
        class r(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return (np.array([1], dtype=int), [[1, 2]], [[1]])
        r_pol = r_policy(r())
        with self.assertRaises(RuntimeError) as cm:
            r_pol.replace(([1, 2], [[.1, .2], [.3, .4]], [
                [1.1], [2.2]]), 2, 0, 1, 0, 0, [], ([], [], []))

        # Test that construction from another pygmo.r_policy fails.
        with self.assertRaises(TypeError) as cm:
            r_policy(r_pol)
        err = cm.exception
        self.assertTrue(
            "a pygmo.r_policy cannot be used as a UDRP for another pygmo.r_policy (if you need to copy a replacement policy please use the standard Python copy()/deepcopy() functions)" in str(err))

    def run_extract_tests(self):
        from .core import r_policy, _test_r_policy, fair_replace
        import sys

        # First we try with a C++ test r_pol.
        t = r_policy(_test_r_policy())
        # Verify the refcount of p is increased after extract().
        rc = sys.getrefcount(t)
        tr_pol = t.extract(_test_r_policy)
        self.assertEqual(sys.getrefcount(t), rc + 1)
        del tr_pol
        self.assertEqual(sys.getrefcount(t), rc)
        # Verify we are modifying the inner object.
        t.extract(_test_r_policy).set_n(5)
        self.assertEqual(t.extract(_test_r_policy).get_n(), 5)

        class tr_policy(object):

            def __init__(self):
                self._n = 1

            def get_n(self):
                return self._n

            def set_n(self, n):
                self._n = n

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return inds

        # Test with Python r_policy.
        t = r_policy(tr_policy())
        rc = sys.getrefcount(t)
        tr_pol = t.extract(tr_policy)
        # Reference count does not increase because
        # tr_policy is stored as a proper Python object
        # with its own refcount.
        self.assertTrue(sys.getrefcount(t) == rc)
        self.assertTrue(tr_pol.get_n() == 1)
        tr_pol.set_n(12)
        self.assert_(t.extract(tr_policy).get_n() == 12)

        # Check that we can extract Python UDTs also via Python's object type.
        t = r_policy(tr_policy())
        self.assertTrue(not t.extract(object) is None)
        # Check we are referring to the same object.
        self.assertEqual(id(t.extract(object)), id(t.extract(tr_policy)))
        # Check that it will not work with exposed C++ replacement policies.
        t = r_policy(fair_replace())
        self.assertTrue(t.extract(object) is None)
        self.assertTrue(not t.extract(fair_replace) is None)

    def run_name_info_tests(self):
        from .core import r_policy

        class t(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return inds

        r_pol = r_policy(t())
        self.assertTrue(r_pol.get_name() != '')
        self.assertTrue(r_pol.get_extra_info() == '')

        class t(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return inds

            def get_name(self):
                return 'pippo'

        r_pol = r_policy(t())
        self.assertTrue(r_pol.get_name() == 'pippo')
        self.assertTrue(r_pol.get_extra_info() == '')

        class t(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return inds

            def get_extra_info(self):
                return 'pluto'

        r_pol = r_policy(t())
        self.assertTrue(r_pol.get_name() != '')
        self.assertTrue(r_pol.get_extra_info() == 'pluto')

        class t(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return inds

            def get_name(self):
                return 'pippo'

            def get_extra_info(self):
                return 'pluto'

        r_pol = r_policy(t())
        self.assertTrue(r_pol.get_name() == 'pippo')
        self.assertTrue(r_pol.get_extra_info() == 'pluto')

    def run_pickle_tests(self):
        from .core import r_policy, fair_replace
        from pickle import dumps, loads
        t_ = r_policy(fair_replace())
        t = loads(dumps(t_))
        self.assertEqual(repr(t), repr(t_))
        self.assertTrue(t.is_(fair_replace))

        t_ = r_policy(_r_pol())
        t = loads(dumps(t_))
        self.assertEqual(repr(t), repr(t_))
        self.assertTrue(t.is_(_r_pol))
