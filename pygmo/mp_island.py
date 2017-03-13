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
# The context manager functionality for the multiprocessing modules is
# available since Python 3.4.
_has_context = _sys.version_info[0] > 3 or (
    _sys.version_info[0] == 3 and _sys.version_info[1] >= 4)


class _temp_disable_sigint(object):
    # A small helper context class to disable CTRL+C while evolution
    # is ongoing in a separate process.

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


def _evolve_func(q, algo, pop):
    # The evolve function that is actually run from the separate processes.
    with _temp_disable_sigint():
        try:
            newpop = algo.evolve(pop)
            q.put(newpop)
        except BaseException as e:
            q.put(e)


class mp_island(object):

    def __init__(self, method=None):
        if not _has_context and method is not None:
            raise ValueError(
                'the "method" parameter must be None in Python < 3.4')
        if method is not None and not isinstance(method, str):
            raise TypeError(
                'the "method" parameter must be either None or a string')
        self._method = method

    def run_evolve(self, algo, pop):
        import multiprocessing as mp
        if _has_context:
            ctx = mp.get_context(self._method)
            q = ctx.Queue()
            p = ctx.Process(target=_evolve_func, args=(q, algo, pop))
        else:
            q = mp.Queue()
            p = mp.Process(target=_evolve_func, args=(q, algo, pop))
        p.start()
        retval = q.get()
        p.join()
        if isinstance(retval, BaseException):
            raise retval
        return retval

    def get_name(self):
        return "Multiprocessing island"
