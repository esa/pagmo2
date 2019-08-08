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

from .core import topology


def _topology_extract(self, t):
    """Extract the user-defined topology.

    This method allows to extract a reference to the user-defined topology (UDT) stored within this
    :class:`~pygmo.topology` instance. The behaviour of this function depends on the value
    of *t* (which must be a :class:`type`) and on the type of the internal UDT:

    * if the type of the UDT is *t*, then a reference to the UDT will be returned
      (this mirrors the behaviour of the corresponding C++ method
      :cpp:func:`pagmo::topology::extract()`),
    * if *t* is :class:`object` and the UDT is a Python object (as opposed to an
      :ref:`exposed C++ topology <py_topologies_cpp>`), then a reference to the
      UDT will be returned (this allows to extract a Python UDT without knowing its type),
    * otherwise, :data:`None` will be returned.

    Args:
        t (:class:`type`): the type of the user-defined topology to extract

    Returns:
        a reference to the internal user-defined topology, or :data:`None` if the extraction fails

    Raises:
        TypeError: if *t* is not a :class:`type`

    Examples:
        >>> import pygmo as pg
        >>> t1 = pg.topology(pg.ring())
        >>> t1.extract(pg.ring) # doctest: +SKIP
        <pygmo.core.ring at 0x7f8e4792b670>
        >>> t1.extract(pg.unconnected) is None
        True
        >>> class topo:
        ...     def get_connections(self, n):
        ...         return [[], []]
        ...     def push_back(self):
        ...         return
        >>> t2 = pg.topology(topo())
        >>> t2.extract(object) # doctest: +SKIP
        <__main__.topo at 0x7f8e478c04e0>
        >>> t2.extract(topo) # doctest: +SKIP
        <__main__.topo at 0x7f8e478c04e0>
        >>> t2.extract(pg.unconnected) is None
        True

    """
    if not isinstance(t, type):
        raise TypeError("the 't' parameter must be a type")
    if hasattr(t, "_pygmo_cpp_topology"):
        return self._cpp_extract(t())
    return self._py_extract(t)


def _topology_is(self, t):
    """Check the type of the user-defined topology.

    This method returns :data:`False` if :func:`extract(t) <pygmo.topology.extract>` returns
    :data:`None`, and :data:`True` otherwise.

    Args:
        t (:class:`type`): the type that will be compared to the type of the UDT

    Returns:
        bool: whether the UDT is of type *t* or not

    Raises:
        unspecified: any exception thrown by :func:`~pygmo.topology.extract()`

    """
    return not self.extract(t) is None


# Do the actual patching.
setattr(topology, "extract", _topology_extract)
setattr(topology, "is_", _topology_is)
