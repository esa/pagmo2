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

from threading import Lock as _Lock


def _mp_ipy_bfe_func(ser_prob_dv):
    # The function that will be invoked
    # by the individual processes/nodes of mp/ipy bfe.
    import pickle

    prob = pickle.loads(ser_prob_dv[0])
    dv = pickle.loads(ser_prob_dv[1])

    return pickle.dumps(prob.fitness(dv))


class mp_bfe(object):
    # Static variables for the pool.
    _pool_lock = _Lock()
    _pool = None
    _pool_size = None

    def __init__(self, chunksize=None):
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

        return fvs

    def get_name(self):
        return "Multiprocessing batch fitness evaluator"

    def get_extra_info(self):
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
        with mp_bfe._pool_lock:
            mp_bfe._init_pool_impl(processes)

    @staticmethod
    def get_pool_size():
        with mp_bfe._pool_lock:
            mp_bfe._init_pool_impl(None)
            return mp_bfe._pool_size

    @staticmethod
    def resize_pool(processes):
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
        with mp_bfe._pool_lock:
            if mp_bfe._pool is not None:
                mp_bfe._pool.close()
                mp_bfe._pool.join()
                mp_bfe._pool = None
                mp_bfe._pool_size = None


class ipyparallel_bfe(object):
    # Static variables for the view.
    _view_lock = _Lock()
    _view = None

    @staticmethod
    def init_view(client_args=[], client_kwargs={}, view_args=[], view_kwargs={}):
        from ipyparallel import Client
        import gc

        # Create the new view.
        new_view = Client(
            *client_args, **client_kwargs).load_balanced_view(*view_args, **view_kwargs)

        with ipyparallel_bfe._view_lock:
            if not ipyparallel_bfe._view is None:
                # Erase the old view.
                old_view = ipyparallel_bfe._view
                ipyparallel_bfe._view = None
                del(old_view)
                gc.collect()

            # Assign the new view.
            ipyparallel_bfe._view = new_view

    @staticmethod
    def shutdown_view():
        import gc
        with ipyparallel_bfe._view_lock:
            if ipyparallel_bfe._view is None:
                return

            old_view = ipyparallel_bfe._view
            ipyparallel_bfe._view = None
            del(old_view)
            gc.collect()

    def __call__(self, prob, dvs):
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

        return fvs

    def get_name(self):
        return "Ipyparallel batch fitness evaluator"

    def get_extra_info(self):
        from copy import deepcopy
        with ipyparallel_bfe._view_lock:
            if ipyparallel_bfe._view is None:
                return "\tNo cluster view has been created yet"
            else:
                d = deepcopy(ipyparallel_bfe._view.queue_status())
        return "\tQueue status:\n\t\n\t" + "\n\t".join(["(" + str(k) + ", " + str(d[k]) + ")" for k in d])
