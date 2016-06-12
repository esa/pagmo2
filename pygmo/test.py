# -*- coding: iso-8859-1 -*-

from __future__ import absolute_import as _ai

import unittest as _ut

class doctests_test_case(_ut.TestCase):
	"""Test case that will run all the doctests.

	"""
	def runTest(self):
		import doctest
		import pygmo
		doctest.testmod(pygmo)

class core_test_case(_ut.TestCase):
	"""Test case for core PyGMO functionality.

	"""
	def runTest(self):
		import sys
		from numpy import random, all
		from .core import _builtin, _type, _str, _callable, _deepcopy, _test_object_serialization as tos
		if sys.version_info[0] < 3:
			import __builtin__ as b
		else:
			import builtins as b
		self.assertEqual(b,_builtin())
		self.assertEqual(type(int),_type(int))
		self.assertEqual(str(123),_str(123))
		self.assertEqual(callable(1),_callable(1))
		self.assertEqual(callable(lambda _: None),_callable(lambda _: None))
		l = [1,2,3,["abc"]]
		self.assert_(id(l) != id(_deepcopy(l)))
		self.assert_(id(l[3]) != id(_deepcopy(l)[3]))
		self.assertEqual(tos(l),l)
		self.assertEqual(tos({'a':l,3:"Hello world"}),{'a':l,3:"Hello world"})
		a = random.rand(3,2)
		self.assert_(all(tos(a) == a))

class problem_test_case(_ut.TestCase):
	"""Test case for the :class:`~pygmo.core.problem` class.

	"""
	def runTest(self):
		from .core import problem
		# First a few non-problems.
		self.assertRaises(TypeError,lambda : problem(1))
		self.assertRaises(TypeError,lambda : problem("hello world"))
		self.assertRaises(TypeError,lambda : problem([]))
		self.assertRaises(TypeError,lambda : problem(int))
		# Some problems missing methods.
		class np0(object):
			def fitness(self,a):
				return [1]
		self.assertRaises(TypeError,lambda : problem(np0))
		class np1(object):
			def get_bounds(self):
				return ([0],[1])
		self.assertRaises(TypeError,lambda : problem(np1))
		# The minimal good citizen.
		glob = []
		class p(object):
			def __init__(self,g):
				self.g = g
			def get_bounds(self):
				return ([0,0],[1,1])
			def fitness(self,a):
				self.g.append(1)
				return [42]
		p_inst = p(glob)
		prob = problem(p_inst)
		# Check a few problem properties.
		self.assertEqual(prob.get_nobj(),1)
		self.assert_(all(prob.get_bounds()[0] == [0,0]))
		self.assertEqual(prob.get_nx(),2)
		self.assert_(all(prob.get_bounds()[1] == [1,1]))
		self.assertEqual(prob.get_nf(),1)
		self.assertEqual(prob.get_nec(),0)
		self.assertEqual(prob.get_nic(),0)
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
		self.assert_(all(prob.fitness([0,0]) == [42]))
		# Run fitness a few more times.
		prob.fitness([0,0])
		prob.fitness([0,0])
		# Assert that the global variable was copied into p, not simply referenced.
		self.assertEqual(len(glob),0)
		self.assertEqual(len(prob.extract(p).g),3)

def run_test_suite():
	"""Run the full test suite.

	This function will raise an exception if at least one test fails.

	"""
	retval = 0
	suite = _ut.TestLoader().loadTestsFromTestCase(doctests_test_case)
	suite.addTest(problem_test_case())
	suite.addTest(core_test_case())
	test_result = _ut.TextTestRunner(verbosity=2).run(suite)
	if len(test_result.failures) > 0:
		retval = 1
	if retval != 0:
		raise RuntimeError('One or more tests failed.')
