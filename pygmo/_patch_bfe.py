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


def _bfe_call(self, prob, dvs):
    """Call operator.

    The call operator will invoke the internal UDBFE instance to perform the evaluation in batch mode
    of the decision vectors stored in *dvs* using the input :class:`~pygmo.problem` or UDP *prob*,
    and it will return the corresponding fitness vectors.

    The input decision vectors must be stored contiguously in *dvs*: for a problem with dimension :math:`n`, the first
    decision vector in *dvs* occupies the index range :math:`\left[0, n\right)`, the second decision vector
    occupies the range :math:`\left[n, 2n\right)`, and so on. Similarly, the output fitness vectors must be
    laid out contiguously in the return value: for a problem with fitness dimension :math:`f`, the first fitness
    vector will occupy the index range :math:`\left[0, f\right)`, the second fitness vector
    will occupy the range :math:`\left[f, 2f\right)`, and so on.

    This function will perform a variety of sanity checks on both *dvs* and on the return value.

    Args:
        prob (:class:`~pygmo.problem` or a UDP): the input problem
        dvs (array-like object): the input decision vectors that will be evaluated in batch mode

    Returns:
        1D NumPy float array: the fitness vectors corresponding to the input decision vectors in *dvs*

    Raises:
        ValueError: if *dvs* or the return value produced by the UDBFE are incompatible with the input problem *prob*
        unspecified: any exception raised by the invocation of the UDBFE, or by failures at the intersection
          between C++ and Python (e.g., type conversion errors, mismatched function signatures, etc.)

    """
    from .core import problem
    if isinstance(prob, problem):
        return self._call_impl(prob, dvs)
    else:
        return self._call_impl(problem(prob), dvs)


# Do the actual patching.
setattr(bfe, "extract", _bfe_extract)
setattr(bfe, "is_", _bfe_is)
setattr(bfe, "__call__", _bfe_call)
