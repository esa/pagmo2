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

from .core import s_policy


def _s_policy_extract(self, t):
    """Extract the user-defined selection policy.

    This method allows to extract a reference to the user-defined selection policy (UDSP) stored within this
    :class:`~pygmo.s_policy` instance. The behaviour of this function depends on the value
    of *t* (which must be a :class:`type`) and on the type of the internal UDSP:

    * if the type of the UDSP is *t*, then a reference to the UDSP will be returned
      (this mirrors the behaviour of the corresponding C++ method
      :cpp:func:`pagmo::s_policy::extract()`),
    * if *t* is :class:`object` and the UDSP is a Python object (as opposed to an
      :ref:`exposed C++ selection policy <py_s_policies_cpp>`), then a reference to the
      UDSP will be returned (this allows to extract a Python UDSP without knowing its type),
    * otherwise, :data:`None` will be returned.

    Args:
        t (:class:`type`): the type of the user-defined selection policy to extract

    Returns:
        a reference to the internal user-defined selection policy, or :data:`None` if the extraction fails

    Raises:
        TypeError: if *t* is not a :class:`type`

    """
    if not isinstance(t, type):
        raise TypeError("the 't' parameter must be a type")
    if hasattr(t, "_pygmo_cpp_s_policy"):
        return self._cpp_extract(t())
    return self._py_extract(t)


def _s_policy_is(self, t):
    """Check the type of the user-defined selection policy.

    This method returns :data:`False` if :func:`extract(t) <pygmo.s_policy.extract>` returns
    :data:`None`, and :data:`True` otherwise.

    Args:
        t (:class:`type`): the type that will be compared to the type of the UDSP

    Returns:
        bool: whether the UDSP is of type *t* or not

    Raises:
        unspecified: any exception thrown by :func:`~pygmo.s_policy.extract()`

    """
    return not self.extract(t) is None


# Do the actual patching.
setattr(s_policy, "extract", _s_policy_extract)
setattr(s_policy, "is_", _s_policy_is)
