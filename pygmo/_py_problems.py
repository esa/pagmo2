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


def _with_decorator(f):
    from functools import wraps

    @wraps(f)
    def wrapper(self, *args, **kwds):
        dec = self._decors.get(f.__name__)
        if dec is None:
            return f(self, *args, **kwds)
        else:
            return dec(f)(self, *args, **kwds)
    return wrapper


class decorator_problem(object):
    """Decorator meta-problem.


    """

    def __init__(self, prob, **kwargs):
        from . import problem
        from warnings import warn
        from copy import deepcopy
        if type(prob) == problem:
            # If prob is a pygmo problem, we will make a copy
            # and store it. The copy is to ensure consistent behaviour
            # with the other meta problems and with the constructor
            # from a UDP (which will end up making a deep copy of
            # the input object).
            self._prob = deepcopy(prob)
        else:
            # Otherwise, we attempt to create a problem from it. This will
            # work if prob is an exposed C++ problem or a Python UDP.
            self._prob = problem(prob)
        self._decors = {}
        for k in kwargs:
            if k.endswith("_decorator"):
                if not callable(kwargs[k]):
                    raise ValueError(
                        "Cannot register the decorator for the '{}' method: the supplied object "
                        "'{}' is not callable".format(k[:-10], kwargs[k]))
                self._decors[k[:-10]] = deepcopy(kwargs[k])
            else:
                warn("Unrecognized keyword argument: '{}'".format(k))

    @_with_decorator
    def fitness(self, dv):
        return self._prob.fitness(dv)

    @_with_decorator
    def get_bounds(self):
        return self._prob.get_bounds()

    @_with_decorator
    def get_nobj(self):
        return self._prob.get_nobj()

    @_with_decorator
    def get_nec(self):
        return self._prob.get_nec()

    @_with_decorator
    def get_nic(self):
        return self._prob.get_nic()

    @_with_decorator
    def get_nix(self):
        return self._prob.get_nix()

    @_with_decorator
    def has_gradient(self):
        return self._prob.has_gradient()

    @_with_decorator
    def gradient(self, dv):
        return self._prob.gradient(dv)

    @_with_decorator
    def has_gradient_sparsity(self):
        return self._prob.has_gradient_sparsity()

    @_with_decorator
    def gradient_sparsity(self):
        return self._prob.gradient_sparsity()

    @_with_decorator
    def has_hessians(self):
        return self._prob.has_hessians()

    @_with_decorator
    def hessians(self, dv):
        return self._prob.hessians(dv)

    @_with_decorator
    def has_hessians_sparsity(self):
        return self._prob.has_hessians_sparsity()

    @_with_decorator
    def hessians_sparsity(self):
        return self._prob.hessians_sparsity()

    @_with_decorator
    def has_set_seed(self):
        return self._prob.has_set_seed()

    @_with_decorator
    def set_seed(self, s):
        return self._prob.set_seed(s)

    @_with_decorator
    def get_name(self):
        return self._prob.get_name() + " [decorated]"

    @_with_decorator
    def get_extra_info(self):
        retval = self._prob.get_extra_info()
        if len(self._decors) == 0:
            retval += "\tNo registered decorators.\n"
        else:
            retval += "\tRegistered decorators:\n"
            for i, k in enumerate(self._decors):
                retval += "\t\t" + k + \
                    (",\n" if i < len(self._decors) - 1 else "")
            retval += '\n'
        return retval

    @property
    def inner_problem(self):
        return self._prob
