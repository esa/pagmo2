# -*- coding: utf-8 -*-

# Copyright 2017-2020 PaGMO development team
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

from threading import Lock as _Lock


def _mp_ipy_bfe_func(ser_prob_dv):
    # The function that will be invoked
    # by the individual processes/nodes of mp/ipy bfe.
    import pickle

    prob = pickle.loads(ser_prob_dv[0])
    dv = pickle.loads(ser_prob_dv[1])

    return pickle.dumps(prob.fitness(dv))


class mp_bfe(object):
    """Multiprocessing batch fitness evaluator.

    .. versionadded:: 2.13

    This user-defined batch fitness evaluator (UDBFE) will dispatch
    the fitness evaluation in batch mode of a set of decision vectors
    to a process pool created and managed via the facilities of the
    standard Python :mod:`multiprocessing` module.

    The evaluations of the decision vectors are dispatched to the processes
    of a global :class:`pool <multiprocessing.pool.Pool>` shared between
    different instances of :class:`~pygmo.mp_bfe`. The pool is created
    either implicitly by the construction of the first :class:`~pygmo.mp_bfe`
    object or explicitly via the :func:`~pygmo.mp_bfe.init_pool()`
    static method. The default number of processes in the pool is equal to
    the number of logical CPUs on the current machine. The pool's size can
    be queried via :func:`~pygmo.mp_bfe.get_pool_size()`, and changed via
    :func:`~pygmo.mp_bfe.resize_pool()`. The pool can be stopped via
    :func:`~pygmo.mp_bfe.shutdown_pool()`.

    .. note::

       Due to certain implementation details of CPython, it is not possible to initialise, resize or shutdown the pool
       from a thread different from the main one. Normally this is not a problem, but, for instance, if the first
       :class:`~pygmo.mp_bfe` instance is created in a thread different from the main one, an error
       will be raised. In such a situation, the user should ensure to call :func:`~pygmo.mp_bfe.init_pool()`
       from the main thread before spawning the secondary thread.

    .. warning::

       Due to internal limitations of CPython, sending an interrupt signal (e.g., by pressing ``Ctrl+C`` in an interactive
       Python session) while an :class:`~pygmo.mp_bfe` is running might end up sending an interrupt signal also to the
       external process(es). This can lead to unpredictable runtime behaviour (e.g., the session may hang). Although
       pygmo tries hard to limit as much as possible the chances of this occurrence, it cannot eliminate them completely. Users
       are thus advised to tread carefully with interrupt signals (especially in interactive sessions) when using
       :class:`~pygmo.mp_bfe`.

    .. warning::

       Due to an `upstream bug <https://bugs.python.org/issue38501>`_, when using Python 3.8 the multiprocessing
       machinery may lead to a hangup when exiting a Python session. As a workaround until the bug is resolved, users
       are advised to explicitly call :func:`~pygmo.mp_bfe.shutdown_pool()` before exiting a Python session.

    """

    # Static variables for the pool.
    _pool_lock = _Lock()
    _pool = None
    _pool_size = None

    def __init__(self, chunksize=None):
        """
        Args:

           chunksize(:class:`int` or :data:`None`): if not :data:`None`, this positive integral represents
             the approximate number of decision vectors that are processed by each task
             submitted to the process pool by the call operator

        Raises:

           TypeError: if *chunksize* is neither :data:`None` nor a value of an integral type
           ValueError: if *chunksize* is not strictly positive
           unspecified: any exception thrown by :func:`~pygmo.mp_bfe.init_pool()`

        """
        if not chunksize is None and not isinstance(chunksize, int):
            raise TypeError(
                "The 'chunksize' argument must be None or an int, but it is of type '{}' instead".format(type(chunksize)))

        if not chunksize is None and chunksize <= 0:
            raise ValueError(
                "The 'chunksize' parameter must be a positive integer, but its value is {} instead".format(chunksize))

        # Init the process pool, if necessary.
        mp_bfe.init_pool()

        # Save the chunk size parameter.
        self._chunksize = chunksize

    def __call__(self, prob, dvs):
        """Call operator.

        This method will evaluate in batch mode the fitnesses of the input decision vectors
        *dvs* using the fitness function from the optimisation problem *prob*. The fitness
        evaluations are delegated to the processes of the pool backing
        :class:`~pygmo.mp_bfe`.

        See the documentation of :class:`pygmo.bfe` for an explanation of the expected
        formats of *dvs* and of the return value.

        Args:

           prob(:class:`~pygmo.problem`): the input problem
           dvs(:class:`numpy.ndarray`): the input decision vectors, represented as a
             flattened 1D array

        Returns:

           :class:`numpy.ndarray`: the fitness vectors corresponding to *dvs*, represented as a
             flattened 1D array

        Raises:

           unspecified: any exception thrown by the evaluations, by the (de)serialization
             of the input arguments or of the return value, or by the public interface of the
             process pool


        """
        import pickle
        import numpy as np

        # Fetch the dimension and the fitness
        # dimension of the problem.
        ndim = prob.get_nx()
        nf = prob.get_nf()

        # Compute the total number of decision
        # vectors represented by dvs.
        ndvs = len(dvs) // ndim
        # Reshape dvs so that it represents
        # ndvs decision vectors of dimension ndim
        # each.
        dvs.shape = (ndvs, ndim)

        # Pre-serialize the problem.
        pprob = pickle.dumps(prob)

        # Build the list of arguments to pass
        # to the processes in the pool.
        async_args = [(pprob, pickle.dumps(dv)) for dv in dvs]

        with mp_bfe._pool_lock:
            # Make sure the pool exists.
            mp_bfe._init_pool_impl(None)
            # Runt the objfun evaluations in async mode.
            if self._chunksize is None:
                ret = mp_bfe._pool.map_async(_mp_ipy_bfe_func, async_args)
            else:
                ret = mp_bfe._pool.map_async(
                    _mp_ipy_bfe_func, async_args, chunksize=self._chunksize)

        # Build the vector of fitness vectors as a 2D numpy array.
        fvs = np.array([pickle.loads(fv) for fv in ret.get()])
        # Reshape it so that it is 1D.
        fvs.shape = (ndvs*nf,)

        # Ensure the increment the fevals for prob.
        prob.increment_fevals(ndvs)

        return fvs

    def get_name(self):
        """Name of this evaluator.

        Returns:

           :class:`str`: ``"Multiprocessing batch fitness evaluator"``

        """
        return "Multiprocessing batch fitness evaluator"

    def get_extra_info(self):
        """Extra info for this evaluator.

        If the process pool was previously shut down via :func:`~pygmo.mp_bfe.shutdown_pool()`,
        invoking this function will trigger the creation of a new pool.

        Returns:

           :class:`str`: a string containing information about the number of processes in the pool

        Raises:

           unspecified: any exception thrown by :func:`~pygmo.mp_bfe.get_pool_size()`

        """
        return "\tNumber of processes in the pool: {}".format(
            mp_bfe.get_pool_size())

    @staticmethod
    def _init_pool_impl(processes):
        # Implementation method for initing
        # the pool. This will *not* do any locking.
        from ._mp_utils import _make_pool

        if mp_bfe._pool is None:
            mp_bfe._pool, mp_bfe._pool_size = _make_pool(processes)

    @staticmethod
    def init_pool(processes=None):
        """Initialise the process pool.

        This method will initialise the process pool backing :class:`~pygmo.mp_bfe`, if the pool
        has not been initialised yet or if the pool was previously shut down via :func:`~pygmo.mp_bfe.shutdown_pool()`.
        Otherwise, this method will have no effects.

        Args:

           processes(:data:`None` or an :class:`int`): the size of the pool (if :data:`None`, the size of the pool will be
             equal to the number of logical CPUs on the system)

        Raises:

           ValueError: if the pool does not exist yet and the function is being called from a thread different
             from the main one, or if *processes* is a non-positive value
           TypeError: if *processes* is not :data:`None` and not an :class:`int`

        """
        with mp_bfe._pool_lock:
            mp_bfe._init_pool_impl(processes)

    @staticmethod
    def get_pool_size():
        """Get the size of the process pool.

        If the process pool was previously shut down via :func:`~pygmo.mp_bfe.shutdown_pool()`, invoking this
        function will trigger the creation of a new pool.

        Returns:

           :class:`int`: the current size of the pool

        Raises:

           unspecified: any exception thrown by :func:`~pygmo.mp_bfe.init_pool()`

        """
        with mp_bfe._pool_lock:
            mp_bfe._init_pool_impl(None)
            return mp_bfe._pool_size

    @staticmethod
    def resize_pool(processes):
        """Resize pool.

        This method will resize the process pool backing :class:`~pygmo.mp_bfe`.

        If the process pool was previously shut down via :func:`~pygmo.mp_bfe.shutdown_pool()`, invoking this
        function will trigger the creation of a new pool.

        Args:

           processes(:class:`int`): the desired number of processes in the pool

        Raises:

           TypeError: if the *processes* argument is not an :class:`int`
           ValueError: if the *processes* argument is not strictly positive
           unspecified: any exception thrown by :func:`~pygmo.mp_bfe.init_pool()`

        """
        from ._mp_utils import _make_pool

        if not isinstance(processes, int):
            raise TypeError("The 'processes' argument must be an int")
        if processes <= 0:
            raise ValueError(
                "The 'processes' argument must be strictly positive")

        with mp_bfe._pool_lock:
            # NOTE: this will either init a new pool
            # with the requested number of processes,
            # or do nothing if the pool exists already.
            mp_bfe._init_pool_impl(processes)
            if processes == mp_bfe._pool_size:
                # Don't do anything if we are not changing
                # the size of the pool.
                return
            # Create new pool.
            new_pool, new_size = _make_pool(processes)
            # Stop the current pool.
            mp_bfe._pool.close()
            mp_bfe._pool.join()
            # Assign the new pool.
            mp_bfe._pool = new_pool
            mp_bfe._pool_size = new_size

    @staticmethod
    def shutdown_pool():
        """Shutdown pool.

        This method will shut down the process pool backing :class:`~pygmo.mp_bfe`, after
        all pending tasks in the pool have completed.

        After the process pool has been shut down, attempting to use the evaluator
        will raise an error. A new process pool can be created via an explicit call to
        :func:`~pygmo.mp_bfe.init_pool()` or one of the methods of the public API of
        :class:`~pygmo.mp_bfe` which trigger the creation of a new process pool.

        """
        with mp_bfe._pool_lock:
            if mp_bfe._pool is not None:
                mp_bfe._pool.close()
                mp_bfe._pool.join()
                mp_bfe._pool = None
                mp_bfe._pool_size = None


class ipyparallel_bfe(object):
    """Ipyparallel batch fitness evaluator.

    .. versionadded:: 2.13

    This user-defined batch fitness evaluator (UDBFE) will dispatch
    the fitness evaluation in batch mode of a set of decision vectors
    to an ipyparallel cluster. The communication with the cluster is managed
    via an :class:`ipyparallel.LoadBalancedView` instance which is
    created either implicitly when the first fitness evaluation is run, or
    explicitly via the :func:`~pygmo.ipyparallel_bfe.init_view()` method. The
    :class:`~ipyparallel.LoadBalancedView` instance is a global object shared
    among all the ipyparallel batch fitness evaluators.

    .. seealso::

       https://ipyparallel.readthedocs.io/en/latest/

    """
    # Static variables for the view.
    _view_lock = _Lock()
    _view = None

    @staticmethod
    def init_view(client_args=[], client_kwargs={}, view_args=[], view_kwargs={}):
        """Init the ipyparallel view.

        This method will initialise the :class:`ipyparallel.LoadBalancedView`
        which is used by all ipyparallel evaluators to submit the evaluation tasks
        to an ipyparallel cluster. If the :class:`ipyparallel.LoadBalancedView`
        has already been created, this method will perform no action.

        The input arguments *client_args* and *client_kwargs* are forwarded
        as positional and keyword arguments to the construction of an
        :class:`ipyparallel.Client` instance. From the constructed client,
        an :class:`ipyparallel.LoadBalancedView` instance is then created
        via the :func:`ipyparallel.Client.load_balanced_view()` method, to
        which the positional and keyword arguments *view_args* and
        *view_kwargs* are passed.

        Note that usually it is not necessary to explicitly invoke this
        method: an :class:`ipyparallel.LoadBalancedView` is automatically
        constructed with default settings the first time a batch evaluation task
        is submitted to an ipyparallel evaluator. This method should be used
        only if it is necessary to pass custom arguments to the construction
        of the :class:`ipyparallel.Client` or :class:`ipyparallel.LoadBalancedView`
        objects.

        Args:

            client_args(:class:`list`): the positional arguments used for the
              construction of the client
            client_kwargs(:class:`dict`): the keyword arguments used for the
              construction of the client
            view_args(:class:`list`): the positional arguments used for the
              construction of the view
            view_kwargs(:class:`dict`): the keyword arguments used for the
              construction of the view

        Raises:

           unspecified: any exception thrown by the constructor of :class:`ipyparallel.Client`
             or by the :func:`ipyparallel.Client.load_balanced_view()` method

        """
        from ipyparallel import Client
        import gc

        with ipyparallel_bfe._view_lock:
            if ipyparallel_bfe._view is None:
                # Create the new view.
                ipyparallel_bfe._view = Client(
                    *client_args, **client_kwargs).load_balanced_view(*view_args, **view_kwargs)

    @staticmethod
    def shutdown_view():
        """Destroy the ipyparallel view.

        This method will destroy the :class:`ipyparallel.LoadBalancedView`
        currently being used by the ipyparallel evaluators for submitting
        evaluation tasks to an ipyparallel cluster. The view can be re-inited
        implicitly by submitting a new evaluation task, or by invoking
        the :func:`~pygmo.ipyparallel_bfe.init_view()` method.

        """
        import gc
        with ipyparallel_bfe._view_lock:
            if ipyparallel_bfe._view is None:
                return

            old_view = ipyparallel_bfe._view
            ipyparallel_bfe._view = None
            del(old_view)
            gc.collect()

    def __call__(self, prob, dvs):
        """Call operator.

        This method will evaluate in batch mode the fitnesses of the input decision vectors
        *dvs* using the fitness function from the optimisation problem *prob*. The fitness
        evaluations are delegated to the nodes of the ipyparallel cluster backing
        :class:`~pygmo.ipyparallel_bfe`.

        See the documentation of :class:`pygmo.bfe` for an explanation of the expected
        formats of *dvs* and of the return value.

        Args:

           prob(:class:`~pygmo.problem`): the input problem
           dvs(:class:`numpy.ndarray`): the input decision vectors, represented as a
             flattened 1D array

        Returns:

           :class:`numpy.ndarray`: the fitness vectors corresponding to *dvs*, represented as a
             flattened 1D array

        Raises:

           unspecified: any exception thrown by the evaluations, by the (de)serialization
             of the input arguments or of the return value, or by the public interface of
             :class:`ipyparallel.LoadBalancedView`.

        """
        import pickle
        import numpy as np

        # Fetch the dimension and the fitness
        # dimension of the problem.
        ndim = prob.get_nx()
        nf = prob.get_nf()

        # Compute the total number of decision
        # vectors represented by dvs.
        ndvs = len(dvs) // ndim
        # Reshape dvs so that it represents
        # ndvs decision vectors of dimension ndim
        # each.
        dvs.shape = (ndvs, ndim)

        # Pre-serialize the problem.
        pprob = pickle.dumps(prob)

        # Build the list of arguments to pass
        # to the cluster nodes.
        async_args = [(pprob, pickle.dumps(dv)) for dv in dvs]

        with ipyparallel_bfe._view_lock:
            if ipyparallel_bfe._view is None:
                from ipyparallel import Client
                ipyparallel_bfe._view = Client().load_balanced_view()
            ret = ipyparallel_bfe._view.map_async(_mp_ipy_bfe_func, async_args)

        # Build the vector of fitness vectors as a 2D numpy array.
        fvs = np.array([pickle.loads(fv) for fv in ret.get()])
        # Reshape it so that it is 1D.
        fvs.shape = (ndvs*nf,)

        # Ensure the increment the fevals for prob.
        prob.increment_fevals(ndvs)

        return fvs

    def get_name(self):
        """Name of the evaluator.

        Returns:
            :class:`str`: ``"Ipyparallel batch fitness evaluator"``

        """
        return "Ipyparallel batch fitness evaluator"

    def get_extra_info(self):
        """Extra info for this evaluator.

        Returns:
            :class:`str`: a string with extra information about the status of the evaluator

        """
        from copy import deepcopy
        with ipyparallel_bfe._view_lock:
            if ipyparallel_bfe._view is None:
                return "\tNo cluster view has been created yet"
            else:
                d = deepcopy(ipyparallel_bfe._view.queue_status())
        return "\tQueue status:\n\t\n\t" + "\n\t".join(["(" + str(k) + ", " + str(d[k]) + ")" for k in d])
