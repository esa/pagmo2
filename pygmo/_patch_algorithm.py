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

from .core import algorithm


def _algorithm_extract(self, t):
    """Extract the user-defined algorithm.

    This method allows to extract a reference to the user-defined algorithm (UDA) stored within this
    :class:`~pygmo.algorithm` instance. The behaviour of this function depends on the value
    of *t* (which must be a :class:`type`) and on the type of the internal UDA:

    * if the type of the UDA is *t*, then a reference to the UDA will be returned
      (this mirrors the behaviour of the corresponding C++ method
      :cpp:func:`pagmo::algorithm::extract()`),
    * if *t* is :class:`object` and the UDA is a Python object (as opposed to an
      :ref:`exposed C++ algorithm <py_algorithms_cpp>`), then a reference to the
      UDA will be returned (this allows to extract a Python UDA without knowing its type),
    * otherwise, :data:`None` will be returned.

    Args:
        t (:class:`type`): the type of the user-defined algorithm to extract

    Returns:
        a reference to the internal user-defined algorithm, or :data:`None` if the extraction fails

    Raises:
        TypeError: if *t* is not a :class:`type`

    Examples:
        >>> import pygmo as pg
        >>> a1 = pg.algorithm(pg.compass_search())
        >>> a1.extract(pg.compass_search) # doctest: +SKIP
        <pygmo.core.compass_search at 0x7f8e4792b670>
        >>> a1.extract(pg.de) is None
        True
        >>> class algo:
        ...     def evolve(self, pop):
        ...         return pop
        >>> a2 = pg.algorithm(algo())
        >>> a2.extract(object) # doctest: +SKIP
        <__main__.algo at 0x7f8e478c04e0>
        >>> a2.extract(algo) # doctest: +SKIP
        <__main__.algo at 0x7f8e478c04e0>
        >>> a2.extract(pg.de) is None
        True

    """
    if not isinstance(t, type):
        raise TypeError("the 't' parameter must be a type")
    if hasattr(t, "_pygmo_cpp_algorithm"):
        return self._cpp_extract(t())
    return self._py_extract(t)


def _algorithm_is(self, t):
    """Check the type of the user-defined algorithm.

    This method returns :data:`False` if :func:`extract(t) <pygmo.algorithm.extract>` returns
    :data:`None`, and :data:`True` otherwise.

    Args:
        t (:class:`type`): the type that will be compared to the type of the UDA

    Returns:
        bool: whether the UDA is of type *t* or not

    Raises:
        unspecified: any exception thrown by :func:`~pygmo.algorithm.extract()`

    """
    return not self.extract(t) is None


# Do the actual patching.
setattr(algorithm, "extract", _algorithm_extract)
setattr(algorithm, "is_", _algorithm_is)
