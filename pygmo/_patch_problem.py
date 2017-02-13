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

from .core import problem, translate, decompose

# This is the original ctor for problem, defined in C++.
# It will be overridden below.
__original_problem_init = problem.__init__


def _problem_init(self, udp):
    # Override the C++-defined constructor, so that we can
    # record the udp type.

    # Call the original constructor.
    __original_problem_init(self, udp)

    # Construction was succesful: record the UDP type.
    self._udp_type = type(udp)


__original_translate_init = translate.__init__


def _translate_init(self, udp=None, translation=None):
    # Override translate's init, the same way done for problem.
    if udp is None and translation is None:
        __original_translate_init(self)
    else:
        __original_translate_init(self, udp, translation)
    self._udp_type = type(udp)


def _problem_extract(self, t):
    """Extract user-defined problem instance.

    If *t* is the same type of the user-defined problem used to construct this problem, then a reference to
    the internal user-defined problem will be returned. Otherwise, ``None`` will be returned.

    **NOTE**: this functionality is offered to mirror the C++ :cpp:func:`pagmo::problem::extract()` API.
    A more convenient way to access the internal UDP is provided via the :attr:`~pygmo.core.problem.udp`
    property.

    Args:
        t (``type``): the type of the user-defined problem to extract

    Returns:
        a reference to the internal user-defined problem if it is of type *t*, or ``None`` otherwise

    Raises:
        TypeError: if *t* is not a type

    """
    if not isinstance(t, type):
        raise TypeError("the 't' parameter must be a type")
    if hasattr(t, "_pygmo_cpp_problem"):
        return self._cpp_extract(t())
    return self._py_extract(t)


def _problem_is(self, t):
    """Check the type of the user-defined problem instance.

    If *t* is the same type of the user-defined problem used to construct this problem, then ``True`` will be
    returned. Otherwise, ``False`` will be returned.

    **NOTE**: this functionality is offered to mirror the C++ :cpp:func:`pagmo::problem::is()` API.
    A more convenient way to access the internal UDP is provided via the :attr:`~pygmo.core.problem.udp`
    property.

    Args:
        t (type): the type of the user-defined problem to extract

    Returns:
        bool: whether the user-defined problem is of type *t* or not

    Raises:
        TypeError: if *t* is not a type

    """
    return not self.extract(t) is None


@property
def _problem_udp(self):
    """Access the UDP.

    This read-only property grants access to the UDP that was used for the construction of the problem.

    Returns:
        a reference to the internal user-defined problem

    """
    return self.extract(self._udp_type)


@property
def _problem_is_pythonic(self):
    if hasattr(self.udp, "is_pythonic"):
        return self.udp.is_pythonic
    return not hasattr(self._udp_type, "_pygmo_cpp_problem")

# Do the actual patching.
setattr(problem, "__init__", _problem_init)
setattr(problem, "udp", _problem_udp)
setattr(problem, "extract", _problem_extract)
setattr(problem, "is_", _problem_is)
setattr(problem, "is_pythonic", _problem_is_pythonic)
setattr(translate, "__init__", _translate_init)
setattr(translate, "udp", _problem_udp)
setattr(translate, "extract", _problem_extract)
setattr(translate, "is_", _problem_is)
setattr(translate, "is_pythonic", _problem_is_pythonic)
setattr(decompose, "udp", _problem_udp)
setattr(decompose, "extract", _problem_extract)
setattr(decompose, "is_", _problem_is)
