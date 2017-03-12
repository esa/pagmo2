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

from ipyparallel import Client as _Client, use_dill as _use_dill
from threading import Lock as _Lock

# Make sure we use dill for serialization.
_use_dill()


def _evolve_func(algo, pop):
    # This is the actual function that will be sent to the engine.
    return algo.evolve(pop)

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
