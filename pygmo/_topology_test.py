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


class _topo(object):

    def get_connections(self, n):
        return [[], []]

    def push_back(self):
        return


class topology_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.topology` class.

    """

    def runTest(self):
        self.run_basic_tests()
        self.run_extract_tests()
        self.run_name_info_tests()
        self.run_pickle_tests()

    def run_basic_tests(self):
        # Tests for minimal topology, and mandatory methods.
        from numpy import ndarray, dtype
        from .core import topology, ring, unconnected
        # Def construction.
        t = topology()
        self.assertTrue(t.extract(unconnected) is not None)
        self.assertTrue(t.extract(ring) is None)

        # First a few non-topos.
        self.assertRaises(NotImplementedError, lambda: topology(1))
        self.assertRaises(NotImplementedError,
                          lambda: topology("hello world"))
        self.assertRaises(NotImplementedError, lambda: topology([]))
        self.assertRaises(TypeError, lambda: topology(int))
        # Some topologies missing methods, wrong arity, etc.

        class nt0(object):
            pass
        self.assertRaises(NotImplementedError, lambda: topology(nt0()))

        class nt1(object):

            get_connections = 45
            push_back = 45
        self.assertRaises(NotImplementedError, lambda: topology(nt1()))

        # The minimal good citizen.
        glob = []

        class t(object):

            def __init__(self, g):
                self.g = g

            def push_back(self):
                self.g.append(1)
                return 1

            def get_connections(self, n):
                self.g.append(2)
                return [[], []]

        t_inst = t(glob)
        topo = topology(t_inst)

        with self.assertRaises(OverflowError) as cm:
            topo.push_back(n=-1)

        # Test the keyword arg.
        topo = topology(udt=ring())
        topo = topology(udt=t_inst)

        # Check a few topo properties.
        self.assertEqual(topo.get_extra_info(), "")
        self.assertTrue(topo.extract(int) is None)
        self.assertTrue(topo.extract(ring) is None)
        self.assertFalse(topo.extract(t) is None)
        self.assertTrue(topo.is_(t))
        self.assertTrue(isinstance(topo.get_connections(0), tuple))
        self.assertTrue(isinstance(topo.get_connections(0)[0], ndarray))
        self.assertTrue(isinstance(topo.get_connections(0)[1], ndarray))
        self.assertTrue(topo.get_connections(0)[1].dtype == dtype(float))
        # Assert that t_inst was deep-copied into topo:
        # the instance in topo will have its own copy of glob
        # and it will not be a reference the outside object.
        self.assertEqual(len(glob), 0)
        self.assertEqual(len(topo.extract(t).g), 4)
        self.assertEqual(topo.extract(t).g, [2]*4)
        self.assertTrue(topo.push_back() is None)
        self.assertEqual(topo.extract(t).g, [2]*4 + [1])

        topo = topology(ring())
        self.assertTrue(topo.get_extra_info() != "")
        self.assertTrue(topo.extract(int) is None)
        self.assertTrue(topo.extract(t) is None)
        self.assertFalse(topo.extract(ring) is None)
        self.assertTrue(topo.is_(ring))
        self.assertTrue(isinstance(topo.push_back(), type(None)))

        # Wrong retval for get_connections().

        class t(object):

            def push_back(self):
                pass

            def get_connections(self, n):
                return []
        topo = topology(t())
        self.assertRaises(RuntimeError, lambda: topo.get_connections(0))

        class t(object):

            def push_back(self):
                pass

            def get_connections(self, n):
                return [1]
        topo = topology(t())
        self.assertRaises(RuntimeError, lambda: topo.get_connections(0))

        class t(object):

            def push_back(self):
                pass

            def get_connections(self, n):
                return [1, 2, 3]
        topo = topology(t())
        self.assertRaises(RuntimeError, lambda: topo.get_connections(0))

        class t(object):

            def push_back(self):
                pass

            def get_connections(self, n):
                return [[1, 2, 3], [.5]]
        topo = topology(t())
        with self.assertRaises(ValueError) as cm:
            topo.get_connections(0)
        err = cm.exception
        self.assertTrue(
            "while the vector of migration probabilities has a size of" in str(err))

        class t(object):

            def push_back(self):
                pass

            def get_connections(self, n):
                return [[1, 2, 3], [.5, .6, 1.4]]
        topo = topology(t())
        with self.assertRaises(ValueError) as cm:
            topo.get_connections(0)
        err = cm.exception
        self.assertTrue(
            "An invalid migration probability of " in str(err))

        class t(object):

            def push_back(self):
                pass

            def get_connections(self, n):
                return [[1, 2, 3], [.5, .6, float("inf")]]
        topo = topology(t())
        with self.assertRaises(ValueError) as cm:
            topo.get_connections(0)
        err = cm.exception
        self.assertTrue(
            "An invalid non-finite migration probability of " in str(err))

        # Test that construction from another pygmo.topology fails.
        with self.assertRaises(TypeError) as cm:
            topology(topo)
        err = cm.exception
        self.assertTrue(
            "a pygmo.topology cannot be used as a UDT for another pygmo.topology (if you need to copy a topology please use the standard Python copy()/deepcopy() functions)" in str(err))

    def run_extract_tests(self):
        from .core import topology, _test_topology, ring
        import sys

        # First we try with a C++ test topo.
        t = topology(_test_topology())
        # Verify the refcount of p is increased after extract().
        rc = sys.getrefcount(t)
        ttopo = t.extract(_test_topology)
        self.assertEqual(sys.getrefcount(t), rc + 1)
        del ttopo
        self.assertEqual(sys.getrefcount(t), rc)
        # Verify we are modifying the inner object.
        t.extract(_test_topology).set_n(5)
        self.assertEqual(t.extract(_test_topology).get_n(), 5)

        class ttopology(object):

            def __init__(self):
                self._n = 1

            def get_n(self):
                return self._n

            def set_n(self, n):
                self._n = n

            def get_connections(self, n):
                return [[], []]

            def push_back(self):
                pass

        # Test with Python topology.
        t = topology(ttopology())
        rc = sys.getrefcount(t)
        ttopo = t.extract(ttopology)
        # Reference count does not increase because
        # ttopology is stored as a proper Python object
        # with its own refcount.
        self.assertTrue(sys.getrefcount(t) == rc)
        self.assertTrue(ttopo.get_n() == 1)
        ttopo.set_n(12)
        self.assert_(t.extract(ttopology).get_n() == 12)

        # Check that we can extract Python UDTs also via Python's object type.
        t = topology(ttopology())
        self.assertTrue(not t.extract(object) is None)
        # Check we are referring to the same object.
        self.assertEqual(id(t.extract(object)), id(t.extract(ttopology)))
        # Check that it will not work with exposed C++ topologies.
        t = topology(ring())
        self.assertTrue(t.extract(object) is None)
        self.assertTrue(not t.extract(ring) is None)

    def run_name_info_tests(self):
        from .core import topology

        class t(object):

            def get_connections(self, n):
                return [[], []]

            def push_back(self):
                pass

        topo = topology(t())
        self.assertTrue(topo.get_name() != '')
        self.assertTrue(topo.get_extra_info() == '')

        class t(object):

            def get_connections(self, n):
                return [[], []]

            def push_back(self):
                pass

            def get_name(self):
                return 'pippo'

        topo = topology(t())
        self.assertTrue(topo.get_name() == 'pippo')
        self.assertTrue(topo.get_extra_info() == '')

        class t(object):

            def get_connections(self, n):
                return [[], []]

            def push_back(self):
                pass

            def get_extra_info(self):
                return 'pluto'

        topo = topology(t())
        self.assertTrue(topo.get_name() != '')
        self.assertTrue(topo.get_extra_info() == 'pluto')

        class t(object):

            def get_connections(self, n):
                return [[], []]

            def push_back(self):
                pass

            def get_name(self):
                return 'pippo'

            def get_extra_info(self):
                return 'pluto'

        topo = topology(t())
        self.assertTrue(topo.get_name() == 'pippo')
        self.assertTrue(topo.get_extra_info() == 'pluto')

    def run_pickle_tests(self):
        from .core import topology, ring
        from pickle import dumps, loads
        t_ = topology(ring())
        t = loads(dumps(t_))
        self.assertEqual(repr(t), repr(t_))
        self.assertTrue(t.is_(ring))

        t_ = topology(_topo())
        t = loads(dumps(t_))
        self.assertEqual(repr(t), repr(t_))
        self.assertTrue(t.is_(_topo))
