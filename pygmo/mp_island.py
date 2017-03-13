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
import sys as _sys
import threading as _thr

# The context manager functionality for the multiprocessing module is
# available since Python 3.4.
_has_context = _sys.version_info[0] > 3 or (
    _sys.version_info[0] == 3 and _sys.version_info[1] >= 4)


class _temp_disable_sigint(object):
    # A small helper context class to disable CTRL+C temporarily.

    def __enter__(self):
        import signal
        # Record the previous sigint handler.
        self._prev_signal = signal.getsignal(signal.SIGINT)
        # Assign the new sig handler (i.e., ignore SIGINT).
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def __exit__(self, type, value, traceback):
        import signal
        # Restore the previous sighandler.
        signal.signal(signal.SIGINT, self._prev_signal)

_pool_lock = _thr.Lock()
_pool = None
_pool_size = None


def _evolve_func(algo, pop):
    # The evolve function that is actually run from the separate processes.
    return algo.evolve(pop)


class mp_island(object):

    def __init__(self, method=None, processes=None):
        import multiprocessing as mp
        if not _has_context and method is not None:
            raise ValueError(
                'the "method" parameter must be None in Python < 3.4')
        if method is not None and not isinstance(method, str):
            raise TypeError(
                'the "method" parameter must be either None or a string')
        # TODO check processes.
        global _pool, _pool_size

        def create_pool():
            with _temp_disable_sigint():
                if _has_context:
                    ctx = mp.get_context(method)
                    _pool = ctx.Pool(processes)
                else:
                    _pool = mp.Pool(processes)
            _pool_size = processes
        with _pool_lock:
            if _pool is None:
                create_pool()
            elif processes != _pool_size:
                _pool.close()
                _pool.join()
                create_pool()

    def run_evolve(self, algo, pop):
        with _pool_lock:
            res = _pool.apply_async(_evolve_func, (algo, pop))
        return res.get()

    def get_name(self):
        return "Multiprocessing island"
