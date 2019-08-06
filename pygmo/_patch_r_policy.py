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

from .core import r_policy


def _r_policy_extract(self, t):
    """Extract the user-defined replacement policy.

    This method allows to extract a reference to the user-defined replacement policy (UDRP) stored within this
    :class:`~pygmo.r_policy` instance. The behaviour of this function depends on the value
    of *t* (which must be a :class:`type`) and on the type of the internal UDRP:

    * if the type of the UDRP is *t*, then a reference to the UDRP will be returned
      (this mirrors the behaviour of the corresponding C++ method
      :cpp:func:`pagmo::r_policy::extract()`),
    * if *t* is :class:`object` and the UDRP is a Python object (as opposed to an
      :ref:`exposed C++ replacement policy <py_r_policies_cpp>`), then a reference to the
      UDRP will be returned (this allows to extract a Python UDRP without knowing its type),
    * otherwise, :data:`None` will be returned.

    Args:
        t (:class:`type`): the type of the user-defined replacement policy to extract

    Returns:
        a reference to the internal user-defined replacement policy, or :data:`None` if the extraction fails

    Raises:
        TypeError: if *t* is not a :class:`type`

    """
    if not isinstance(t, type):
        raise TypeError("the 't' parameter must be a type")
    if hasattr(t, "_pygmo_cpp_r_policy"):
        return self._cpp_extract(t())
    return self._py_extract(t)


def _r_policy_is(self, t):
    """Check the type of the user-defined replacement policy.

    This method returns :data:`False` if :func:`extract(t) <pygmo.r_policy.extract>` returns
    :data:`None`, and :data:`True` otherwise.

    Args:
        t (:class:`type`): the type that will be compared to the type of the UDRP

    Returns:
        bool: whether the UDRP is of type *t* or not

    Raises:
        unspecified: any exception thrown by :func:`~pygmo.r_policy.extract()`

    """
    return not self.extract(t) is None


# Do the actual patching.
setattr(r_policy, "extract", _r_policy_extract)
setattr(r_policy, "is_", _r_policy_is)
