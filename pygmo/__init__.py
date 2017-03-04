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

# for python 2.0 compatibility
from __future__ import absolute_import as _ai

# We import the sub-modules into the root namespace
from .core import *
from .plotting import *

# And we explicitly import the submodules
from . import core
from . import plotting
from . import test

# Patch the problem class.
from . import _patch_problem


# Patch the algorithm class.
from . import _patch_algorithm


class thread_safety(object):
    """Thread safety level.

    This enum defines a set of values that can be used to specify the thread safety of problems, algorithms, etc.

    """

    #: No thread safety: concurrent operations on distinct instances are unsafe
    none = core._thread_safety.none
    #: Basic thread safety: concurrent operations on distinct instances are safe
    basic = core._thread_safety.basic


# Override of the population constructor.
__original_population_init = population.__init__


def _population_init(self, prob=None, size=0, seed=None):
    # NOTE: the idea of having the pop init here instead of exposed from C++ is that like this we don't need
    # to expose a new pop ctor each time we expose a new problem: in this method we will use the problem ctor
    # from a C++ problem, and on the C++ exposition side we need only to
    # expose the ctor of pop from pagmo::problem.
    """
    Args:
        prob: a user-defined problem (either Python or C++), or an instance of :class:`~pygmo.core.problem`
            (if ``None``, the population problem will be :class:`~pygmo.core.null_problem`)
        size (``int``): the number of individuals
        seed (``int``): the random seed (if ``None``, it will be randomly-generated)

    Raises:
        TypeError: if *size* is not an ``int`` or *seed* is not ``None`` and not an ``int``
        OverflowError:  is *size* or *seed* are negative
        unspecified: any exception thrown by the invoked C++ constructors or by the constructor of
            :class:`~pygmo.core.problem`, or by failures at the intersection between C++ and
            Python (e.g., type conversion errors, mismatched function signatures, etc.)

    """
    import sys
    int_types = (int, long) if sys.version_info[0] < 3 else (int,)
    # Check input params.
    if not isinstance(size, int_types):
        raise TypeError("the 'size' parameter must be an integer")
    if not seed is None and not isinstance(seed, int_types):
        raise TypeError("the 'seed' parameter must be None or an integer")
    if prob is None:
        # Use the null problem for default init.
        prob = null_problem()
    if type(prob) == problem:
        # If prob is a pygmo problem, we will pass it as-is to the
        # original init.
        prob_arg = prob
    else:
        # Otherwise, we attempt to create a problem from it. This will
        # work if prob is an exposed C++ problem or a Python UDP.
        prob_arg = problem(prob)
    if seed is None:
        __original_population_init(self, prob_arg, size)
    else:
        __original_population_init(self, prob_arg, size, seed)

setattr(population, "__init__", _population_init)

# Register the cleanup function.
import atexit as _atexit
from .core import _cleanup
_atexit.register(lambda: _cleanup())
