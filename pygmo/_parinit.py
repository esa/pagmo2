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

from threading import Lock as _Lock

_mp_pool = None
_mp_pool_size = None
_mp_pool_lock = _Lock()


def _fitness_f(prob, dv):
    return prob.fitness(dv)


def _mp_fitness_dispatch(prob, dv):
    from ._mp_utils import _make_pool

    global _mp_pool
    global _mp_pool_size
    global _mp_pool_lock

    with _mp_pool_lock:
        if _mp_pool is None:
            _mp_pool, _mp_pool_size = _make_pool(None)
        return _mp_pool.apply_async(_fitness_f, (prob, dv))


def _cleanup():
    global _mp_pool
    global _mp_pool_size
    global _mp_pool_lock

    with _mp_pool_lock:
        if _mp_pool is not None:
            _mp_pool.close()
            _mp_pool.join()
            _mp_pool = None
            _mp_pool_size = None
