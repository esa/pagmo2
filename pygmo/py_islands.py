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

from threading import Lock as _Lock
from ipyparallel import Client as _Client, use_dill as _use_dill


def _evolve_func(algo, pop):
    # The evolve function that is actually run from the separate processes
    # in both mp_island and ipyparallel_island.
    return algo.evolve(pop)


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


class mp_island(object):
    _pool_lock = _Lock()
    _pool = None
    _pool_size = None

    def __init__(self):
        # Init the process pool, if necessary.
        mp_island.init_pool()

    def run_evolve(self, algo, pop):
        with mp_island._pool_lock:
            res = mp_island._pool.apply_async(_evolve_func, (algo, pop))
        return res.get()

    def get_name(self):
        return "Multiprocessing island"

    @staticmethod
    def init_pool(processes=None):
        # Helper to create a new pool. It will do something
        # only if the pool has never been created before.
        import multiprocessing as mp
        import sys
        # The mp island requires Python 3.4 at least.
        if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 4):
            raise RuntimeError(
                "The multiprocessing island is supported only for Python >= 3.4.")
        if processes is not None and not isinstance(processes, int):
            raise TypeError("The 'processes' argument must be None or an int")
        if processes is not None and processes <= 0:
            raise ValueError(
                "The 'processes' argument, if not None, must be strictly positive")
        with mp_island._pool_lock:
            if mp_island._pool is None:
                with _temp_disable_sigint():
                    ctx = mp.get_context("spawn")
                    mp_island._pool = ctx.Pool(processes=processes)
                mp_island._pool_size = mp.cpu_count() if processes is None else processes

    @staticmethod
    def get_pool_size():
        mp_island.init_pool()
        with mp_island._pool_lock:
            return mp_island._pool_size

    @staticmethod
    def resize_pool(processes):
        import multiprocessing as mp
        mp_island.init_pool()
        if not isinstance(processes, int):
            raise TypeError("The 'processes' argument must be an int")
        if processes <= 0:
            raise ValueError(
                "The 'processes' argument must be strictly positive")
        with mp_island._pool_lock:
            if processes == mp_island._pool_size:
                # Don't do anything if we are not changing
                # the size of the pool.
                return
            # Create new pool.
            with _temp_disable_sigint():
                ctx = mp.get_context("spawn")
                new_pool = ctx.Pool(processes)
            # Stop the current pool.
            mp_island._pool.close()
            mp_island._pool.join()
            # Assign the new pool.
            mp_island._pool = new_pool
            mp_island._pool_size = processes

    @staticmethod
    def _shutdown_pool():
        with mp_island._pool_lock:
            if mp_island._pool is not None:
                mp_island._pool.close()
                mp_island._pool.join()

# Make sure we use dill for serialization.
_use_dill()


# NOTE: the idea here is that we don't want to create a new client for
# every island: creation is expensive, and we cannot have too many clients
# as after a certain threshold ipyparallel starts erroring out.
# So we store the clients as values in a dict whose keys are the arguments
# passed to Client() upon construction, and we re-use existing clients
# if the construction arguments are identical.
# NOTE: this is not a proper cache as it never kicks anything out, but as
# a temporary solution it is fine. Consider using something like a LRU
# cache in the future.
_client_cache = {}
_client_cache_lock = _Lock()


def _hashable(v):
    # Determine whether v can be hashed.
    try:
        hash(v)
    except TypeError:
        return False
    return True


class ipyparallel_island(object):

    def __init__(self, *args, **kwargs):
        # Turn the arguments into something that might be hashable.
        args_key = (args, tuple(sorted([(k, kwargs[k]) for k in kwargs])))
        if _hashable(args_key):
            with _client_cache_lock:
                if args_key in _client_cache:
                    self._rc = _client_cache[args_key]
                else:
                    _client_cache[args_key] = _Client(*args, **kwargs)
                    self._rc = _client_cache[args_key]
        else:
            # If the arguments are not hashable, just create a brand new
            # client.
            self._rc = _Client(*args, **kwargs)

        # Init the load balanced view.
        self._lview = self._rc.load_balanced_view()

    def __copy__(self):
        # For copy and deepcopy just return a reference to itself,
        # so the copy is not really deep. But it does not make any sense
        # anyway to try to deep copy a connection object.
        return self

    def __deepcopy__(self, d):
        return self

    def run_evolve(self, algo, pop):
        return self._lview.apply_sync(_evolve_func, algo, pop)

    def get_name(self):
        return "Ipyparallel island"
