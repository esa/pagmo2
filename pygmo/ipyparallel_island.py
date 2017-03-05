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


class DelayedKeyboardInterrupt(object):

    def __enter__(self):
        import signal
        self.signal_received = False
        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        import logging
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        import signal
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


class ipyparallel_island(object):

    def __init__(self, pop):
        from ipyparallel import Client
        from copy import deepcopy
        import threading
        from concurrent.futures import ThreadPoolExecutor

        # Store the pop as a data member, with associated lock.
        self._pop = deepcopy(pop)
        self._pop_lock = threading.Lock()

        # Setup of the ipyparallel bits.
        self._rc = Client()
        self._lview = self._rc.load_balanced_view()
        self._lview.block = False
        self._lview_lock = threading.Lock()

        # List of futures.
        self._fut_list = []
        self._fut_list_lock = threading.Lock()

        # Thread executor.
        self._te = ThreadPoolExecutor(max_workers=1)

    def __copy__(self):
        return ipyparallel_island(self.get_population())

    def __deepcopy__(self, d):
        return ipyparallel_island(self.get_population())

    def get_population(self):
        from copy import deepcopy
        with self._pop_lock:
            retval = deepcopy(self._pop)
        return retval

    def set_population(self, pop):
        from copy import deepcopy
        cpop = deepcopy(pop)
        with self._pop_lock:
            self._pop = cpop

    def wait(self):
        exc = None
        with self._fut_list_lock:
            for f in self._fut_list:
                try:
                    f.result()
                except Exception as e:
                    if exc is None:
                        exc = e
            self._fut_list = []
        if exc is not None:
            raise exc

    def enqueue_evolution(self, algo, archi):
        def evolve_func(algo, pop):
            return algo.evolve(pop)

        def thread_func(self, algo, archi):
            with self._lview_lock:
                ar = self._lview.apply(
                    evolve_func, algo, self.get_population())
            self.set_population(ar.get())

        with self._fut_list_lock:
            self._fut_list.append(self._te.submit(
                thread_func, self, algo, archi))
