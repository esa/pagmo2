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
from .py_islands import *

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

    #: No thread safety: any concurrent operation on distinct instances is unsafe
    none = core._thread_safety.none
    #: Copy-only thread safety: concurrent copying of distinct instances is safe
    copyonly = core._thread_safety.copyonly
    #: Basic thread safety: any concurrent operation on distinct instances is safe
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

# Override of the island constructor.
__original_island_init = island.__init__


def _island_init(self, **kwargs):
    """
    Keyword Args:
        algo: a user-defined algorithm (either Python or C++), or an instance of :class:`~pygmo.core.algorithm`
        prob: a user-defined problem (either Python or C++), or an instance of :class:`~pygmo.core.problem`
        udi: a user-defined island (either Python or C++)
        size (``int``): the number of individuals
        seed (``int``): the random seed (if not specified, it will be randomly-generated)

    Raises:
        TypeError: if *size* is not an ``int`` or *seed* is not an ``int``
        ValueError: if the number of keyword arguments is not one of [0, 2, 3, 4]
        KeyError: if the names of the keyword arguments are not consistent with the number of keyword arguments
        OverflowError:  is *size* or *seed* are negative
        unspecified: any exception thrown by the invoked C++ constructors or by the constructors of
            :class:`~pygmo.core.problem` and :class:`~pygmo.core.algorithm`, or by failures at the intersection
            between C++ and Python (e.g., type conversion errors, mismatched function signatures, etc.)

    """
    import sys
    int_types = (int, long) if sys.version_info[0] < 3 else (int,)
    if len(kwargs) == 0:
        __original_island_init(self)
    elif len(kwargs) in [2, 3, 4]:
        if not "algo" in kwargs:
            raise KeyError(
                "the mandatory 'algo' parameter is missing from the list of arguments "
                "of the island constructor")
        algo = kwargs.pop("algo")
        algo = algo if type(algo) == algorithm else algorithm(algo)
        if len(kwargs) == 1:
            if not "udi" in kwargs:
                raise KeyError(
                    "the 'udi' parameter is missing from the 2-arguments "
                    "form of the island constructor. The argument '{}' "
                    "is present instead".format(list(kwargs.keys())[0]))
            __original_island_init(self, algo, kwargs.pop("udi"))
        else:
            keys = sorted(kwargs.keys())
            if keys != ["prob", "seed", "size"] and keys != ["prob", "size"]:
                raise KeyError("one or more parameters are missing from the 3/4-arguments "
                               "form of the island constructor. The 'prob' and 'size' arguments are "
                               "mandatory, the 'seed' argument is optional")
            prob = kwargs.pop("prob")
            prob = prob if type(prob) == problem else problem(prob)
            size = kwargs.pop("size")
            if not isinstance(size, int_types):
                raise TypeError("the 'size' parameter must be an integer")
            if len(kwargs) == 0:
                __original_island_init(self, algo, prob, size)
            else:
                seed = kwargs.pop("seed")
                if not isinstance(seed, int_types):
                    raise TypeError("the 'seed' parameter must be an integer")
                __original_island_init(self, algo, prob, size, seed)
    else:
        raise ValueError(
            "the number of keyword arguments for the island constructor must be "
            "either 0, 2, 3 or 4, but {} arguments were passed instead".format(len(kwargs)))

#setattr(island, "__init__", _island_init)

# Register the cleanup function.
import atexit as _atexit
from .core import _cleanup as _cpp_cleanup


def _cleanup():
    mp_island._shutdown_pool()
    _cpp_cleanup()


_atexit.register(_cleanup)
