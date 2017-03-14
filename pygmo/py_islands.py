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


def _evolve_func(algo, pop):
    # The evolve function that is actually run from the separate processes
    # in both mp_island and ipyparallel_island.
    return algo.evolve(pop)


class _temp_disable_sigint(object):
    # A small helper context class to disable CTRL+C temporarily.

    def __enter__(self):
        import signal
        # Store the previous sigint handler.
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
            # NOTE: run this while the pool is locked. We have
            # functions to modify the pool (e.g., resize()) and
            # we need to make sure we are not trying to touch
            # the pool while we are sending tasks to it.
            res = mp_island._pool.apply_async(_evolve_func, (algo, pop))
        return res.get()

    def get_name(self):
        return "Multiprocessing island"

    @staticmethod
    def _make_pool(processes):
        # A small private factory function to create the a process pool.
        # It accomplishes the tasks of selecting the correct method for
        # starting the processes ("spawn") and making sure that the
        # created processes will ignore the SIGINT signal (this prevents)
        # troubles when the user issues an interruption with ctrl+c from
        # the main process.
        import sys
        import os
        import multiprocessing as mp
        # The context functionality in the mp module is available since
        # Python 3.4. It is uses to force the process creation with the
        # "spawn" method.
        has_context = sys.version_info[0] > 3 or (
            sys.version_info[0] == 3 and sys.version_info[1] >= 4)
        with _temp_disable_sigint():
            # NOTE: we temporarily disable sigint while creating the pool.
            # This ensures that the processes created in the pool will ignore
            # interruptions issued via ctrl+c (only the main process will
            # be affected by them).
            if has_context:
                ctx = mp.get_context("spawn")
                pool = ctx.Pool(processes=processes)
            else:
                # NOTE: for Python < 3.4, only Windows is supported and we
                # should never end up here.
                assert(os.name == 'nt')
                pool = mp.Pool(processes=processes)
        pool_size = mp.cpu_count() if processes is None else processes
        # Return the created pool and its size.
        return pool, pool_size

    @staticmethod
    def init_pool(processes=None):
        # Helper to create a new pool. It will do something
        # only if the pool has never been created before.
        import sys
        import os
        # The mp island requires either Windows or at least Python 3.4.
        if os.name != 'nt' and (sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 4)):
            raise RuntimeError(
                "The multiprocessing island is supported only on Windows or on Python >= 3.4.")
        if processes is not None and not isinstance(processes, int):
            raise TypeError("The 'processes' argument must be None or an int")
        if processes is not None and processes <= 0:
            raise ValueError(
                "The 'processes' argument, if not None, must be strictly positive")
        with mp_island._pool_lock:
            if mp_island._pool is None:
                mp_island._pool, mp_island._pool_size = mp_island._make_pool(
                    processes)

    @staticmethod
    def get_pool_size():
        mp_island.init_pool()
        with mp_island._pool_lock:
            return mp_island._pool_size

    @staticmethod
    def resize_pool(processes):
        import multiprocessing as mp
        if not isinstance(processes, int):
            raise TypeError("The 'processes' argument must be an int")
        if processes <= 0:
            raise ValueError(
                "The 'processes' argument must be strictly positive")
        mp_island.init_pool()
        with mp_island._pool_lock:
            if processes == mp_island._pool_size:
                # Don't do anything if we are not changing
                # the size of the pool.
                return
            # Create new pool.
            new_pool, new_size = mp_island._make_pool(processes)
            # Stop the current pool.
            mp_island._pool.close()
            mp_island._pool.join()
            # Assign the new pool.
            mp_island._pool = new_pool
            mp_island._pool_size = new_size

    @staticmethod
    def _shutdown_pool():
        # This is used only during the shutdown phase of the pygmo module.
        with mp_island._pool_lock:
            if mp_island._pool is not None:
                mp_island._pool.close()
                mp_island._pool.join()


# Make sure we use dill for serialization, if ipyparallel is available.
try:
    from ipyparallel import use_dill as _use_dill
    _use_dill()
except ImportError:
    pass


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
        from ipyparallel import Client
        # Turn the arguments into something that might be hashable.
        args_key = (args, tuple(sorted([(k, kwargs[k]) for k in kwargs])))
        if _hashable(args_key):
            with _client_cache_lock:
                if args_key in _client_cache:
                    self._rc = _client_cache[args_key]
                else:
                    _client_cache[args_key] = Client(*args, **kwargs)
                    self._rc = _client_cache[args_key]
        else:
            # If the arguments are not hashable, just create a brand new
            # client.
            self._rc = Client(*args, **kwargs)

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
        # NOTE: no need to lock, as there's no other way to interact
        # with lview apart from this method.
        return self._lview.apply_sync(_evolve_func, algo, pop)

    def get_name(self):
        return "Ipyparallel island"
