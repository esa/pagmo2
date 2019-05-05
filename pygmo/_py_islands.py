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


def _evolve_func_mp_pool(ser_algo_pop):
    # The evolve function that is actually run from the separate processes
    # in mp_island (when using the pool).
    import pickle
    algo, pop = pickle.loads(ser_algo_pop)
    new_pop = algo.evolve(pop)
    return pickle.dumps((algo, new_pop))


def _evolve_func_mp_pipe(conn, ser_algo_pop):
    # The evolve function that is actually run from the separate processes
    # in mp_island (when *not* using the pool). Communication with the
    # parent process happens through the conn pipe.
    from ._mp_utils import _temp_disable_sigint

    # NOTE: disable SIGINT with the goal of preventing the user from accidentally
    # interrupting the evolution via hitting Ctrl+C in an interactive session
    # in the parent process. Note that this disables the signal only during
    # evolution, but the signal is still enabled when the process is bootstrapping
    # (so the user can still cause troubles in the child process with a well-timed
    # Ctrl+C). There's nothing we can do about it: the only way would be to disable
    # SIGINT before creating the child process, but unfortunately the creation
    # of a child process happens in a separate thread and Python disallows messing
    # with signal handlers from a thread different from the main one :(
    with _temp_disable_sigint():
        import pickle
        try:
            algo, pop = pickle.loads(ser_algo_pop)
            new_pop = algo.evolve(pop)
            conn.send(pickle.dumps((algo, new_pop)))
        except Exception as e:
            conn.send(RuntimeError(
                "An exception was raised in the evolution of a multiprocessing island. The full error message is:\n{}".format(e)))
        finally:
            conn.close()


def _evolve_func_ipy(algo, pop):
    # The evolve function that is actually run from the separate processes
    # in ipyparallel_island.
    new_pop = algo.evolve(pop)
    return algo, new_pop


class mp_island(object):
    """Multiprocessing island.

    .. versionadded:: 2.10

       The *use_pool* parameter (in previous versions, :class:`~pygmo.mp_island` always used a process pool).

    This user-defined island (UDI) will dispatch evolution tasks to an external Python process
    using the facilities provided by the standard Python :mod:`multiprocessing` module.

    If the construction argument *use_pool* is :data:`True`, then a process from a global
    :class:`pool <multiprocessing.pool.Pool>` shared between different instances of
    :class:`~pygmo.mp_island` will be used. The pool is created either implicitly by the construction
    of the first :class:`~pygmo.mp_island` object or explicitly via the :func:`~pygmo.mp_island.init_pool()`
    static method. The default number of processes in the pool is equal to the number of logical CPUs on the
    current machine. The pool's size can be queried via :func:`~pygmo.mp_island.get_pool_size()`,
    and changed via :func:`~pygmo.mp_island.resize_pool()`. The pool can be stopped via
    :func:`~pygmo.mp_island.shutdown_pool()`.

    If *use_pool* is :data:`False`, each evolution launched by an :class:`~pygmo.mp_island` will be offloaded
    to a new :class:`process <multiprocessing.Process>` which will then be terminated at the end of the evolution.

    Generally speaking, a process pool will be faster (and will use fewer resources) than spawning a new process
    for every evolution. A process pool, however, by its very nature limits the number of evolutions that can
    be run simultaneously on the system, and it introduces a serializing behaviour that might not be desirable
    in certain situations (e.g., when studying parallel evolution with migration in an :class:`~pygmo.archipelago`).

    .. note::

       This island type is supported only on Windows or if the Python version is at least 3.4. Attempting to use
       this class on non-Windows platforms with a Python version earlier than 3.4 will raise an error.

    .. note::

       Due to certain implementation details of CPython, it is not possible to initialise, resize or shutdown the pool
       from a thread different from the main one. Normally this is not a problem, but, for instance, if the first
       :class:`~pygmo.mp_island` instance is created in a thread different from the main one, an error
       will be raised. In such a situation, the user should ensure to call :func:`~pygmo.mp_island.init_pool()`
       from the main thread before spawning the secondary thread.

    .. warning::

       Due to internal limitations of CPython, sending an interrupt signal (e.g., by pressing ``Ctrl+C`` in an interactive
       Python session) while an :class:`~pygmo.mp_island` is evolving might end up sending an interrupt signal also to the
       external evolution process(es). This can lead to unpredictable runtime behaviour (e.g., the session may hang). Although
       pygmo tries hard to limit as much as possible the chances of this occurrence, it cannot eliminate them completely. Users
       are thus advised to tread carefully with interrupt signals (especially in interactive sessions) when using
       :class:`~pygmo.mp_island`.

    """

    # Static variables for the pool.
    _pool_lock = _Lock()
    _pool = None
    _pool_size = None

    def __init__(self, use_pool=True):
        """
        Args:

           use_pool(bool): if :data:`True`, a process from a global pool will be used to run the evolution, otherwise a new
              process will be spawned for each evolution

        Raises:

           TypeError: is *use_pool* is not of type :class:`bool`
           RuntimeError: if the multiprocessing island is not supported on the current platform and *use_pool* is :data:`False`
           unspecified: any exception thrown by :func:`~pygmo.mp_island.init_pool()` if *use_pool* is :data:`True`

        """
        self._init(use_pool)

    def _init(self, use_pool):
        # Implementation of the ctor. Factored out
        # because it's re-used in the pickling support.
        if not isinstance(use_pool, bool):
            raise TypeError(
                "The 'use_pool' parameter in the mp_island constructor must be a boolean, but it is of type {} instead.".format(type(use_pool)))
        self._use_pool = use_pool
        if self._use_pool:
            # Init the process pool, if necessary.
            mp_island.init_pool()
        else:
            # Init the pid member and associated lock.
            self._pid_lock = _Lock()
            self._pid = None

    @property
    def use_pool(self):
        """Pool usage flag (read-only).

        Returns:

           bool: :data:`True` if this island uses a process pool, :data:`False` otherwise

        """
        return self._use_pool

    def __copy__(self):
        # For copy/deepcopy, construct a new instance
        # with the same arguments used to construct self.
        # NOTE: no need for locking, as _use_pool is set
        # on construction and never touched again.
        return mp_island(self._use_pool)

    def __deepcopy__(self, d):
        return self.__copy__()

    def __getstate__(self):
        # For pickle/unpickle, we employ the construction
        # argument, which will be used to re-init the class
        # during unpickle.
        return self._use_pool

    def __setstate__(self, state):
        # NOTE: we need to do a full init of the object,
        # in order to set the use_pool flag and, if necessary,
        # construct the _pid and _pid_lock objects.
        self._init(state)

    def run_evolve(self, algo, pop):
        """Evolve population.

        This method will evolve the input :class:`~pygmo.population` *pop* using the input
        :class:`~pygmo.algorithm` *algo*, and return *algo* and the evolved population. The evolution
        is run either on one of the processes of the pool backing :class:`~pygmo.mp_island`, or in
        a new separate process. If this island is using a pool, and the pool was previously
        shut down via :func:`~pygmo.mp_island.shutdown_pool()`, an exception will be raised.

        Args:

           algo(:class:`~pygmo.algorithm`): the input algorithm
           pop(:class:`~pygmo.population`): the input population

        Returns:

           tuple: a tuple of 2 elements containing *algo* (i.e., the :class:`~pygmo.algorithm` object that was used for the evolution) and the evolved :class:`~pygmo.population`

        Raises:

           RuntimeError: if the pool was manually shut down via :func:`~pygmo.mp_island.shutdown_pool()`
           unspecified: any exception thrown during the evolution, or by the public interface of the
             process pool


        """
        # NOTE: the idea here is that we pass the *already serialized*
        # arguments to the mp machinery, instead of letting the multiprocessing
        # module do the serialization. The advantage of doing so is
        # that if there are serialization errors, we catch them early here rather
        # than failing in the bootstrap phase of the remote process, which
        # can lead to hangups.
        import pickle
        ser_algo_pop = pickle.dumps((algo, pop))

        if self._use_pool:
            with mp_island._pool_lock:
                # NOTE: run this while the pool is locked. We have
                # functions to modify the pool (e.g., resize()) and
                # we need to make sure we are not trying to touch
                # the pool while we are sending tasks to it.
                if mp_island._pool is None:
                    raise RuntimeError(
                        "The multiprocessing island pool was stopped. Please restart it via mp_island.init_pool().")
                res = mp_island._pool.apply_async(
                    _evolve_func_mp_pool, (ser_algo_pop,))
            # NOTE: there might be a bug in need of a workaround lurking in here:
            # http://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
            # Just keep it in mind.
            return pickle.loads(res.get())
        else:
            from ._mp_utils import _get_spawn_context

            # Get the context for spawning the process.
            mp_ctx = _get_spawn_context()

            parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
            p = mp_ctx.Process(target=_evolve_func_mp_pipe,
                               args=(child_conn, ser_algo_pop))
            p.start()
            with self._pid_lock:
                self._pid = p.pid
            # NOTE: after setting the pid, wrap everything
            # in a try block with a finally clause for
            # resetting the pid to None. This way, even
            # if there are exceptions, we are sure the pid
            # is set back to None.
            try:
                res = parent_conn.recv()
                p.join()
            finally:
                with self._pid_lock:
                    self._pid = None
            if isinstance(res, RuntimeError):
                raise res
            return pickle.loads(res)

    @property
    def pid(self):
        """ID of the evolution process (read-only).

        This property is available only if the island is *not* using a process pool.

        Returns:

           int: the ID of the process running the current evolution, or :data:`None` if no evolution is ongoing

        Raises:

           ValueError: if the island is using a process pool

        """
        if self._use_pool:
            raise ValueError(
                "The 'pid' property is available only when the island is configured to spawn new processes, but this mp_island is using a process pool instead.")
        with self._pid_lock:
            pid = self._pid
        return pid

    def get_name(self):
        """Island's name.

        Returns:

           str: ``"Multiprocessing island"``

        """
        return "Multiprocessing island"

    def get_extra_info(self):
        """Island's extra info.

        If the island uses a process pool and the pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`,
        invoking this function will trigger the creation of a new pool.

        Returns:

           str: a string containing information about the state of the island (e.g., number of processes in the pool, ID of the evolution process, etc.)

        Raises:

           unspecified: any exception thrown by :func:`~pygmo.mp_island.get_pool_size()`

        """
        retval = "\tUsing a process pool: {}\n".format(
            "yes" if self._use_pool else "no")
        if self._use_pool:
            retval += "\tNumber of processes in the pool: {}".format(
                mp_island.get_pool_size())
        else:
            with self._pid_lock:
                pid = self._pid
            if pid is None:
                retval += "\tNo active evolution process"
            else:
                retval += "\tEvolution process ID: {}".format(pid)
        return retval

    @staticmethod
    def init_pool(processes=None):
        """Initialise the process pool.

        This method will initialise the process pool backing :class:`~pygmo.mp_island`, if the pool
        has not been initialised yet or if the pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`.
        Otherwise, this method will have no effects.

        Args:

           processes(:data:`None` or an :class:`int`): the size of the pool (if :data:`None`, the size of the pool will be
             equal to the number of logical CPUs on the system)

        Raises:

           ValueError: if the pool does not exist yet and the function is being called from a thread different
             from the main one, or if *processes* is a non-positive value
           RuntimeError: if the current platform or Python version is not supported
           TypeError: if *processes* is not :data:`None` and not an :class:`int`

        """
        from ._mp_utils import _make_pool

        with mp_island._pool_lock:
            if mp_island._pool is None:
                mp_island._pool, mp_island._pool_size = _make_pool(processes)

    @staticmethod
    def get_pool_size():
        """Get the size of the process pool.

        If the process pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`, invoking this
        function will trigger the creation of a new pool.

        Returns:

           int: the current size of the pool

        Raises:

           unspecified: any exception thrown by :func:`~pygmo.mp_island.init_pool()`

        """
        mp_island.init_pool()
        with mp_island._pool_lock:
            return mp_island._pool_size

    @staticmethod
    def resize_pool(processes):
        """Resize pool.

        This method will resize the process pool backing :class:`~pygmo.mp_island`.

        If the process pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`, invoking this
        function will trigger the creation of a new pool.

        Args:

           processes(int): the desired number of processes in the pool

        Raises:

           TypeError: if the *processes* argument is not an :class:`int`
           ValueError: if the *processes* argument is not strictly positive
           unspecified: any exception thrown by :func:`~pygmo.mp_island.init_pool()`

        """
        from ._mp_utils import _make_pool

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
            new_pool, new_size = _make_pool(processes)
            # Stop the current pool.
            mp_island._pool.close()
            mp_island._pool.join()
            # Assign the new pool.
            mp_island._pool = new_pool
            mp_island._pool_size = new_size

    @staticmethod
    def shutdown_pool():
        """Shutdown pool.

        .. versionadded:: 2.8

        This method will shut down the process pool backing :class:`~pygmo.mp_island`, after
        all pending tasks in the pool have completed.

        After the process pool has been shut down, attempting to run an evolution on the island
        will raise an error. A new process pool can be created via an explicit call to
        :func:`~pygmo.mp_island.init_pool()` or one of the methods of the public API of
        :class:`~pygmo.mp_island` which trigger the creation of a new process pool.

        """
        with mp_island._pool_lock:
            if mp_island._pool is not None:
                mp_island._pool.close()
                mp_island._pool.join()
                mp_island._pool = None
                mp_island._pool_size = None


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
        # Make sure the kwargs are sorted so that two sets of identical
        # kwargs will be recognized as equal also if the keys are stored
        # in different order.
        args_key = (args, tuple(sorted([(k, kwargs[k]) for k in kwargs])))
        if _hashable(args_key):
            with _client_cache_lock:
                # Try to see if a client constructed with the same
                # arguments already exists in the cache.
                rc = _client_cache.get(args_key)
                if rc is None:
                    # No cached client exists. Create a new client
                    # and store it in the cache.
                    rc = Client(*args, **kwargs)
                    _client_cache[args_key] = rc
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
        :class:`~pygmo.algorithm` *algo*, and return *algo* and the evolved population. The evolution
        task is submitted to the ipyparallel cluster via an internal :class:`ipyparallel.LoadBalancedView`
        instance initialised during the construction of the island.

        Args:

            pop(:class:`~pygmo.population`): the input population
            algo(:class:`~pygmo.algorithm`): the input algorithm

        Returns:

            tuple: a tuple of 2 elements containing *algo* (i.e., the :class:`~pygmo.algorithm` object that was used for the evolution) and the evolved :class:`~pygmo.population`

        Raises:

            unspecified: any exception thrown during the evolution, or by submitting the evolution task
              to the ipyparallel cluster


        """
        with self._view_lock:
            ret = self._lview.apply_async(_evolve_func_ipy, algo, pop)
        return ret.get()

    def get_name(self):
        """Island's name.

        Returns:
            str: ``"Ipyparallel island"``

        """
        return "Ipyparallel island"

    def get_extra_info(self):
        """Island's extra info.

        Returns:
            str: a string with extra information about the status of the island

        """
        with self._view_lock:
            d = self._lview.queue_status()
        return "\tQueue status:\n\t\n\t" + "\n\t".join(["(" + str(k) + ", " + str(d[k]) + ")" for k in d])
