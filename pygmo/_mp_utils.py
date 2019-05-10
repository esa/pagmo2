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


def _get_spawn_context():
    # Small utlity to get a context that will use the 'spawn' method to
    # create new processes with the multiprocessing module. We want to enforce
    # a uniform way of creating new processes across platforms in
    # order to prevent users from implicitly relying on platform-specific
    # behaviour (e.g., fork()), only to discover later that their
    # code is not portable across platforms.
    #
    # The mp context functionality is available from Python 3.4. However,
    # since in Windows the 'spawn' method is the default, we can
    # just return the multiprocessing module in output instead of
    # a context and the 'spawn' method will be used anyway in that case.
    #
    # See:
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

    import multiprocessing as mp
    import os
    import sys

    has_context = sys.version_info[0] > 3 or (
        sys.version_info[0] == 3 and sys.version_info[1] >= 4)
    if has_context:
        mp_ctx = mp.get_context('spawn')
    else:
        if os.name != 'nt':
            raise RuntimeError(
                'Cannot enforce the "spawn" process creation method in Python < 3.4 if we are not on Windows')
        mp_ctx = mp

    return mp_ctx


class _temp_disable_sigint(object):
    # A small helper context class to disable CTRL+C temporarily.

    def __enter__(self):
        import signal
        # Store the previous sigint handler and assign the new sig handler
        # (i.e., ignore SIGINT).
        self._prev_signal = signal.signal(signal.SIGINT, signal.SIG_IGN)

    def __exit__(self, type, value, traceback):
        import signal
        # Restore the previous sighandler.
        signal.signal(signal.SIGINT, self._prev_signal)


def _make_pool(processes):
    # A small factory function to create a process pool.
    # It accomplishes the tasks of selecting the correct method for
    # starting the processes ("spawn") and making sure that the
    # created processes will ignore the SIGINT signal (this prevents
    # troubles when the user issues an interruption with ctrl+c from
    # the main process).

    if processes is not None and not isinstance(processes, int):
        raise TypeError("The 'processes' argument must be None or an int")

    if processes is not None and processes <= 0:
        raise ValueError(
            "The 'processes' argument, if not None, must be strictly positive")

    # Get the context for spawning the process.
    mp_ctx = _get_spawn_context()

    # NOTE: we temporarily disable sigint while creating the pool.
    # This ensures that the processes created in the pool will ignore
    # interruptions issued via ctrl+c (only the main process will
    # be affected by them).
    with _temp_disable_sigint():
        pool = mp_ctx.Pool(processes=processes)

    pool_size = mp_ctx.cpu_count() if processes is None else processes

    # Return the created pool and its size.
    return pool, pool_size
