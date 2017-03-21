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
        udi: a user-defined island (either Python or C++ - note that *udi* will be deep-copied
          and stored inside the :class:`~pygmo.core.island` instance)
        algo: a user-defined algorithm (either Python or C++), or an instance of :class:`~pygmo.core.algorithm`
        pop (:class:`~pygmo.core.population`): a population
        prob: a user-defined problem (either Python or C++), or an instance of :class:`~pygmo.core.problem`
        size (``int``): the number of individuals
        seed (``int``): the random seed (if not specified, it will be randomly-generated)

    Raises:
        KeyError: if the set of keyword arguments is invalid
        unspecified: any exception thrown by:

          * the invoked C++ constructors,
          * the deep copy of the UDI,
          * the constructors of :class:`~pygmo.core.algorithm` and :class:`~pygmo.core.population`,
          * failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
            signatures, etc.)

    """
    if len(kwargs) == 0:
        # Default constructor.
        __original_island_init(self)
        return

    # If we are not dealing with a def ctor, we always need the algo argument.
    if not "algo" in kwargs:
        raise KeyError(
            "the mandatory 'algo' parameter is missing from the list of arguments "
            "of the island constructor")
    algo = kwargs.pop('algo')
    algo = algo if isinstance(algo, algorithm) else algorithm(algo)

    # Population setup. We either need an input pop, or the prob and size,
    # plus optionally seed.
    if 'pop' in kwargs and ('prob' in kwargs or 'size' in kwargs or 'seed' in kwargs):
        raise KeyError(
            "if the 'pop' argument is provided, the 'prob', 'size' and 'seed' "
            "arguments must not be provided")
    elif 'pop' in kwargs:
        pop = kwargs.pop("pop")
    elif 'prob' in kwargs and 'size' in kwargs:
        if 'seed' in kwargs:
            pop = population(kwargs.pop('prob'), kwargs.pop(
                'size'), kwargs.pop('seed'))
        else:
            pop = population(kwargs.pop('prob'), kwargs.pop('size'))
    else:
        raise KeyError(
            "unable to construct a population from the arguments of "
            "the island constructor: you must either pass a population "
            "('pop') or a set of arguments that can be used to build one "
            "('prob', 'size' and, optionally, 'seed')")

    # UDI, if any.
    if 'udi' in kwargs:
        args = [kwargs.pop('udi'), algo, pop]
    else:
        args = [algo, pop]

    if len(kwargs) != 0:
        raise KeyError(
            "unrecognised keyword arguments: {}".format(list(kwargs.keys())))

    __original_island_init(self, *args)


setattr(island, "__init__", _island_init)

# Override of the archi constructor.
__original_archi_init = archipelago.__init__


def _archi_init(self, n=0, **kwargs):
    """
    The keyword arguments accept the same format as explained in the constructor of
    :class:`~pygmo.core.island`. The constructor will initialise an archipelago with
    *n* islands built from *kwargs*.

    Args:
        n (``int``): the number of islands in the archipelago

    Raises:
        TypeError: if *n* is not an integral type
        ValueError: if *n* is negative
        unspecified: any exception thrown by the constructor of :class:`~pygmo.core.island`
          or by the underlying C++ constructor

    """
    import sys
    int_types = (int, long) if sys.version_info[0] < 3 else (int,)
    # Check n.
    if not isinstance(n, int_types):
        raise TypeError("the 'n' parameter must be an integer")
    if n < 0:
        raise ValueError(
            "the 'n' parameter must be non-negative, but it is {} instead".format(n))

    # Call the original init.
    __original_archi_init(self, n, island(**kwargs))

setattr(archipelago, "__init__", _archi_init)


def _archi_push_back(self, **kwargs):
    """Add island.

    This method will construct an island from the supplied arguments and add it to the archipelago.
    Islands are added at the end of the archipelago (that is, the new island will have an index
    equal to the size of the archipelago before the call to this method).

    The keyword arguments accept the same format as explained in the constructor of
    :class:`~pygmo.core.island`.

    Raises:
        unspecified: any exception thrown by the constructor of :class:`~pygmo.core.island` or by
          the underlying C++ method

    """
    self._push_back(island(**kwargs))

setattr(archipelago, "push_back", _archi_push_back)

# Register the cleanup function.
import atexit as _atexit
from .core import _cleanup as _cpp_cleanup


def _cleanup():
    mp_island._shutdown_pool()
    _cpp_cleanup()


_atexit.register(_cleanup)
