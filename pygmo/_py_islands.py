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
        # Store the previous sigint handler and assign the new sig handler
        # (i.e., ignore SIGINT).
        self._prev_signal = signal.signal(signal.SIGINT, signal.SIG_IGN)

    def __exit__(self, type, value, traceback):
        import signal
        # Restore the previous sighandler.
        signal.signal(signal.SIGINT, self._prev_signal)


class mp_island(object):
    """Multiprocessing island.

    This user-defined island (UDI) will dispatch evolution tasks to a pool of processes
    created via the standard Python multiprocessing module. The pool is shared between
    different instances of :class:`~pygmo.mp_island`, and it is created
    either implicitly by the construction of the first :class:`~pygmo.mp_island`
    object or explicitly via the :func:`~pygmo.mp_island.init_pool()` static method.
    The default number of processes in the pool is equal to the number of logical CPUs on the
    current machine. The pool's size can be queried via :func:`~pygmo.mp_island.get_pool_size()`,
    and changed via :func:`~pygmo.mp_island.resize_pool()`.

    .. note::

       Due to certain implementation details of CPython, it is not possible to initialise or resize the pool
       from a thread different from the main one. Normally this is not a problem, but, for instance, if the first
       :class:`~pygmo.mp_island` instance is created in a thread different from the main one, an error
       will be raised. In such a situation, the user should ensure to call :func:`~pygmo.mp_island.init_pool()`
       from the main thread before spawning the secondary thread.

    .. note::

       This island type is supported only on Windows or if the Python version is at least 3.4. Attempting to use
       this class on non-Windows platforms with a Python version earlier than 3.4 will raise an error.

    """
    _pool_lock = _Lock()
    _pool = None
    _pool_size = None

    def __init__(self):
        """
        Raises:

           unspecified: any exception thrown by :func:`~pygmo.mp_island.init_pool()`

        """
        # Init the process pool, if necessary.
        mp_island.init_pool()

    def run_evolve(self, algo, pop):
        """Evolve population.

        This method will evolve the input :class:`~pygmo.population` *pop* using the input
        :class:`~pygmo.algorithm` *algo*, and return the evolved population. The evolution
        is run on one of the processes of the pool backing backing :class:`~pygmo.mp_island`.

        Args:

            pop(:class:`~pygmo.population`): the input population
            algo(:class:`~pygmo.algorithm`): the input algorithm

        Returns:
            :class:`~pygmo.population`: the evolved population

        Raises:
            unspecified: any exception thrown during the evolution, or by the public interface of the
              process pool


        """
        with mp_island._pool_lock:
            # NOTE: run this while the pool is locked. We have
            # functions to modify the pool (e.g., resize()) and
            # we need to make sure we are not trying to touch
            # the pool while we are sending tasks to it.
            res = mp_island._pool.apply_async(_evolve_func, (algo, pop))
        # NOTE: there might be a bug in need of a workaround lurking in here:
        # http://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
        # Just keep it in mind.
        return res.get()

    def get_name(self):
        """Island's name.

        Returns:
            ``str``: ``"Multiprocessing island"``

        """
        return "Multiprocessing island"

    def get_extra_info(self):
        """Island's extra info.

        Returns:
            ``str``: a string specifying the current number of processes in the pool

        """
        return "\tNumber of processes in the pool: {}".format(mp_island.get_pool_size())

    @staticmethod
    def _make_pool(processes):
        # A small private factory function to create a process pool.
        # It accomplishes the tasks of selecting the correct method for
        # starting the processes ("spawn") and making sure that the
        # created processes will ignore the SIGINT signal (this prevents
        # troubles when the user issues an interruption with ctrl+c from
        # the main process).
        import sys
        import os
        import multiprocessing as mp
        # The context functionality in the mp module is available since
        # Python 3.4. It is used to force the process creation with the
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
        """Initialise the process pool.

        This method will initialise the process pool backing :class:`~pygmo.mp_island`, if the pool
        has not been initialised yet. Otherwise, this method will have no effects.

        Args:
            processes(``None`` or an ``int``): the size of the pool (if ``None``, the size of the pool will be
              equal to the number of logical CPUs on the system)

        Raises:

            ValueError: if the pool does not exist yet and the function is being called from a thread different
              from the main one, or if *processes* is a non-positive value
            RuntimeError: if the current platform or Python version is not supported
            TypeError: if *processes* is not ``None`` and not an ``int``

        """
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
        """Get the size of the process pool.

        Returns:

            ``int``: the current size of the pool

        """
        mp_island.init_pool()
        with mp_island._pool_lock:
            return mp_island._pool_size

    @staticmethod
    def resize_pool(processes):
        """Resize pool.

        This method will resize the process pool backing :class:`~pygmo.mp_island`.

        Args:

            processes(``int``): the desired number of processes in the pool

        Raises:

            TypeError: if the *processes* argument is not an ``int``
            ValueError: if the *processes* argument is not strictly positive
            unspecified: any exception thrown by :func:`~pygmo.mp_island.init_pool()`

        """
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


# Make sure we use cloudpickle for serialization, if ipyparallel is available.
try:
    from ipyparallel import use_cloudpickle as _use_cloudpickle
    _use_cloudpickle()
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
    """Ipyparallel island.

    This user-defined island (UDI) will dispatch evolution tasks to an ipyparallel cluster.
    Upon construction, an instance of this UDI will first initialise an :class:`ipyparallel.Client`
    object, and then extract an :class:`ipyparallel.LoadBalancedView` object from it that will
    be used to submit evolution tasks to the cluster. The arguments to the constructor of this
    class will be passed without modifications to the constructor of the :class:`ipyparallel.Client`
    object that will be created internally.

    .. seealso::

       https://ipyparallel.readthedocs.io/en/latest/

    """

    def __init__(self, *args, **kwargs):
        self._lview = self._init(*args, **kwargs)

    def _init(self, *args, **kwargs):
        # A small helper function which will do the following:
        # * get a client from the cache in a thread safe manner, or
        #   create a new one from scratch
        # * store the input arguments as class members
        # * create a LoadBalancedView from the client
        # * create a lock to regulate access to the view
        # * return the view.
        from ipyparallel import Client
        # Turn the arguments into something that might be hashable.
        args_key = (args, tuple(sorted([(k, kwargs[k]) for k in kwargs])))
        if _hashable(args_key):
            with _client_cache_lock:
                if args_key in _client_cache:
                    rc = _client_cache[args_key]
                else:
                    _client_cache[args_key] = Client(*args, **kwargs)
                    rc = _client_cache[args_key]
        else:
            # If the arguments are not hashable, just create a brand new
            # client.
            rc = Client(*args, **kwargs)

        # Save the init arguments.
        self._args = args
        self._kwargs = kwargs

        # NOTE: we need to regulate access to the view because,
        # while run_evolve() is running in a separate thread, we
        # could be doing other things involving the view (e.g.,
        # asking extra_info()). Thus, create the lock here.
        self._view_lock = _Lock()

        return rc.load_balanced_view()

    def __copy__(self):
        # For copy/deepcopy, construct a new instance
        # with the same arguments used to construct self.
        return ipyparallel_island(*self._args, **self._kwargs)

    def __deepcopy__(self, d):
        return self.__copy__()

    def __getstate__(self):
        # For pickle/unpickle, we employ the construction
        # arguments, which will be used to re-init the class
        # during unpickle.
        return self._args, self._kwargs

    def __setstate__(self, state):
        self._lview = self._init(*state[0], **state[1])

    def run_evolve(self, algo, pop):
        """Evolve population.

        This method will evolve the input :class:`~pygmo.population` *pop* using the input
        :class:`~pygmo.algorithm` *algo*, and return the evolved population. The evolution
        task is submitted to the ipyparallel cluster via an internal :class:`ipyparallel.LoadBalancedView`
        instance initialised during the construction of the island.

        Args:

            pop(:class:`~pygmo.population`): the input population
            algo(:class:`~pygmo.algorithm`): the input algorithm

        Returns:
            :class:`~pygmo.population`: the evolved population

        Raises:
            unspecified: any exception thrown during the evolution, or by submitting the evolution task
              to the ipyparallel cluster


        """
        with self._view_lock:
            ret = self._lview.apply_async(_evolve_func, algo, pop)
        return ret.get()

    def get_name(self):
        """Island's name.

        Returns:
            ``str``: ``"Ipyparallel island"``

        """
        return "Ipyparallel island"

    def get_extra_info(self):
        """Island's extra info.

        Returns:
            ``str``: a string with extra information about the status of the island

        """
        with self._view_lock:
            d = self._lview.queue_status()
        return "\tQueue status:\n\t\n\t" + "\n\t".join(["(" + str(k) + ", " + str(d[k]) + ")" for k in d])
