# -*- coding: iso-8859-1 -*-

from __future__ import absolute_import as _ai

import unittest as _ut

class doctests_test_case(_ut.TestCase):
	"""Test case that will run all the doctests.

	To be used within the :mod:`unittest` framework.

	>>> import unittest as ut
	>>> suite = ut.TestLoader().loadTestsFromTestCase(doctests_test_case)

	"""
	def runTest(self):
		import doctest
		import pygmo
		from . import core
		doctest.testmod(pygmo)

def run_test_suite():
	"""Run the full test suite.

	This function will raise an exception if at least one test fails.

	"""
	retval = 0
	suite = _ut.TestLoader().loadTestsFromTestCase(doctests_test_case)
	test_result = _ut.TextTestRunner(verbosity=2).run(suite)
	if len(test_result.failures) > 0:
		retval = 1
	if retval != 0:
		raise RuntimeError('One or more tests failed.')
