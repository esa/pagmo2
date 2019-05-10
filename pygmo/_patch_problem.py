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

# for python 2.0 compatibility
from __future__ import absolute_import as _ai

from .core import problem


def _problem_extract(self, t):
    """Extract the user-defined problem.

    This method allows to extract a reference to the user-defined problem (UDP) stored within this
    :class:`~pygmo.problem` instance. The behaviour of this function depends on the value
    of *t* (which must be a :class:`type`) and on the type of the internal UDP:

    * if the type of the UDP is *t*, then a reference to the UDP will be returned
      (this mirrors the behaviour of the corresponding C++ method
      :cpp:func:`pagmo::problem::extract()`),
    * if *t* is :class:`object` and the UDP is a Python object (as opposed to an
      :ref:`exposed C++ problem <py_problems_cpp>`), then a reference to the
      UDP will be returned (this allows to extract a Python UDP without knowing its type),
    * otherwise, :data:`None` will be returned.

    Args:
        t (:class:`type`): the type of the user-defined problem to extract

    Returns:
        a reference to the internal user-defined problem, or :data:`None` if the extraction fails

    Raises:
        TypeError: if *t* is not a :class:`type`

    Examples:
        >>> import pygmo as pg
        >>> p1 = pg.problem(pg.rosenbrock())
        >>> p1.extract(pg.rosenbrock) # doctest: +SKIP
        <pygmo.core.rosenbrock at 0x7f56b870fd50>
        >>> p1.extract(pg.ackley) is None
        True
        >>> class prob:
        ...     def fitness(self, x):
        ...         return [x[0]]
        ...     def get_bounds(self):
        ...         return ([0],[1])
        >>> p2 = pg.problem(prob())
        >>> p2.extract(object) # doctest: +SKIP
        <__main__.prob at 0x7f56a66b6588>
        >>> p2.extract(prob) # doctest: +SKIP
        <__main__.prob at 0x7f56a66b6588>
        >>> p2.extract(pg.rosenbrock) is None
        True

    """
    if not isinstance(t, type):
        raise TypeError("the 't' parameter must be a type")
    if hasattr(t, "_pygmo_cpp_problem"):
        return self._cpp_extract(t())
    return self._py_extract(t)


def _problem_is(self, t):
    """Check the type of the user-defined problem.

    This method returns :data:`False` if :func:`extract(t) <pygmo.problem.extract>` returns
    :data:`None`, and :data:`True` otherwise.

    Args:
        t (:class:`type`): the type that will be compared to the type of the UDP

    Returns:
        bool: whether the UDP is of type *t* or not

    Raises:
        unspecified: any exception thrown by :func:`~pygmo.problem.extract()`

    """
    return not self.extract(t) is None


# Do the actual patching.
setattr(problem, "extract", _problem_extract)
setattr(problem, "is_", _problem_is)
