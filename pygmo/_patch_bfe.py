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

from .core import bfe


def _bfe_extract(self, t):
    """Extract the user-defined batch fitness evaluator (bfe).

    This method allows to extract a reference to the user-defined bfe (UDBFE) stored within this
    :class:`~pygmo.bfe` instance. The behaviour of this function depends on the value
    of *t* (which must be a :class:`type`) and on the type of the internal UDBFE:

    * if the type of the UDBFE is *t*, then a reference to the UDBFE will be returned
      (this mirrors the behaviour of the corresponding C++ method
      :cpp:func:`pagmo::bfe::extract()`),
    * if *t* is :class:`object` and the UDBFE is a Python object (as opposed to an
      :ref:`exposed C++ bfe <py_bfes_cpp>`), then a reference to the
      UDBFE will be returned (this allows to extract a Python UDBFE without knowing its type),
    * otherwise, :data:`None` will be returned.

    Args:
        t (:class:`type`): the type of the user-defined bfe to extract

    Returns:
        a reference to the internal user-defined bfe, or :data:`None` if the extraction fails

    Raises:
        TypeError: if *t* is not a :class:`type`

    Examples:
        >>> import pygmo as pg
        >>> a1 = pg.bfe(pg.thread_bfe())
        >>> a1.extract(pg.thread_bfe) # doctest: +SKIP
        <pygmo.core.thread_bfe at 0x7f8e4792b670>
        >>> a1.extract(pg.member_bfe) is None
        True
        >>> def custom_bfe(p, dvs): pass
        >>> a2 = pg.bfe(custom_bfe)
        >>> a2.extract(object) # doctest: +SKIP
        <__main__.custom_bfe at 0x7f8e478c04e0>
        >>> a2.extract(custom_bfe) # doctest: +SKIP
        <__main__.custom_bfe at 0x7f8e478c04e0>
        >>> a2.extract(pg.thread_bfe) is None
        True

    """
    if not isinstance(t, type):
        raise TypeError("the 't' parameter must be a type")
    if hasattr(t, "_pygmo_cpp_bfe"):
        return self._cpp_extract(t())
    return self._py_extract(t)


def _bfe_is(self, t):
    """Check the type of the user-defined batch fitness evaluator.

    This method returns :data:`False` if :func:`extract(t) <pygmo.bfe.extract>` returns
    :data:`None`, and :data:`True` otherwise.

    Args:
        t (:class:`type`): the type that will be compared to the type of the UDBFE

    Returns:
        bool: whether the UDBFE is of type *t* or not

    Raises:
        unspecified: any exception thrown by :func:`~pygmo.bfe.extract()`

    """
    return not self.extract(t) is None


# Do the actual patching.
setattr(bfe, "extract", _bfe_extract)
setattr(bfe, "is_", _bfe_is)
