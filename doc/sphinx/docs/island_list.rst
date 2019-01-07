.. _islands:

List of islands (UDIs) available in pagmo/pygmo
================================================

This is the list of user defined islands (UDIs) currently provided with pagmo/pygmo. These are classes that 
can be used to construct a :cpp:class:`pagmo::island` (C++), or a :class:`pygmo.island` (Python), which will then
provide a unified interface to access the island's functionalities.

========================================================== ========================================= =========================================
Common Name                                                Docs of the C++ class                     Docs of the python class                 
========================================================== ========================================= =========================================
Thread island                                              :cpp:class:`pagmo::thread_island`         :class:`pygmo.thread_island`
Fork island                                                :cpp:class:`pagmo::fork_island`           N/A
Multiprocessing island                                     N/A                                       :class:`pygmo.mp_island`
Ipyparallel island                                         N/A                                       :class:`pygmo.ipyparallel_island`
========================================================== ========================================= =========================================
