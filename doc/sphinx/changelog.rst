Changelog
=========

2.12 (unreleased)
-----------------

New
~~~

- The :cpp:class:`pagmo::pso_gen` algorithm can now use the
  batch fitness evaluation scheme
  (`#348 <https://github.com/esa/pagmo2/pull/348>`__).

- Implement the NSPSO algorithm
  (`#314 <https://github.com/esa/pagmo2/pull/314>`__).

Fix
~~~

- Fix a missing ``inline`` and a few wrong include files in the
  serialization header
  (`#355 <https://github.com/esa/pagmo2/pull/355>`__).

- Various build system/doc fixes
  (`#350 <https://github.com/esa/pagmo2/pull/350>`__,
  `#354 <https://github.com/esa/pagmo2/pull/354>`__).

2.11.4 (2019-09-29)
-------------------

Fix
~~~

- Fix an indexing bug in the :cpp:class:`pagmo::pso_gen` algorithm
  (`#349 <https://github.com/esa/pagmo2/pull/349>`__).

- Fix various fitness comparisons when nan values are involved
  (`#346 <https://github.com/esa/pagmo2/pull/346>`__,
  `#347 <https://github.com/esa/pagmo2/pull/347>`__).

2.11.3 (2019-09-09)
-------------------

New
~~~

- :func:`pygmo.archipelago.push_back()` now also accepts :class:`~pygmo.island`
  objects as input arguments (`#342 <https://github.com/esa/pagmo2/pull/342>`__).

Changes
~~~~~~~

- **BREAKING**: the machinery for the translation between C++ and Python
  of vectors of unsigned integral types (e.g., sparsity patterns, individual
  IDs, etc.) now requires that, on the Python side, NumPy arrays are created
  with the appropriate unsigned integral dtype (e.g., ``uint64`` in most
  cases). Previously, pagmo would accept also signed integral dtypes
  (`#342 <https://github.com/esa/pagmo2/pull/342>`__).

Fix
~~~

- Various improvements, fixes and cleanups in the translation of
  C++ vectors to/from Python
  (`#342 <https://github.com/esa/pagmo2/pull/342>`__).

- Fix the printing of islands which contain MO problems
  (`#342 <https://github.com/esa/pagmo2/pull/342>`__).

- Various doc improvements and fixes (`#340 <https://github.com/esa/pagmo2/pull/340>`__).

2.11.2 (2019-08-21)
-------------------

Fix
~~~

- Fix the MinGW pip builds (`#338 <https://github.com/esa/pagmo2/pull/338>`__).

- Fix the default value for the NSGA2 ``eta_m`` parameter in the Python exposition (`#338 <https://github.com/esa/pagmo2/pull/338>`__).

2.11.1 (2019-08-09)
-------------------

Fix
~~~

- Fix a migration issue when multi-objective problems are involved (`#334 <https://github.com/esa/pagmo2/pull/334>`__).

- Various docstring fixes (`#334 <https://github.com/esa/pagmo2/pull/334>`__).

2.11 (2019-08-07)
-----------------

New
~~~

- NSGA2 can optionally use the batch fitness evaluation framework
  (`#308 <https://github.com/esa/pagmo2/pull/308>`__).

- Implement the WFG test suite
  (`#298 <https://github.com/esa/pagmo2/pull/298>`__).

- Migration framework
  (`#296 <https://github.com/esa/pagmo2/pull/296>`__).

- Various additions to the C++ API of user-defined classes
  (`#294 <https://github.com/esa/pagmo2/pull/294>`__).

- Ipopt is now included in the linux pip packages (`#293 <https://github.com/esa/pagmo2/pull/293>`__).

- Implement an ``uninstall`` target in the build system when using the CMake
  ``Unix Makefiles`` generator (`#282 <https://github.com/esa/pagmo2/pull/282>`__).

- Implement the Grey Wolf Optimizer algorithm (`#268 <https://github.com/esa/pagmo2/pull/268>`__).

- Add CircleCI to the continuous integration pipeline (`#266 <https://github.com/esa/pagmo2/pull/266>`__).

- Implement the Extended Ant Colony Optimization algorithm (`#249 <https://github.com/esa/pagmo2/pull/249>`__).

- Implement the Lennard-Jones and Golomb ruler problems (`#247 <https://github.com/esa/pagmo2/pull/247>`__).

- Batch fitness evaluation framework (`#226 <https://github.com/esa/pagmo2/pull/226>`__).

Changes
~~~~~~~

- Various improvements to the MinGW pip packages: the toolchain
  and the dependencies have
  been updated, support for Python 3.7 has been added (`#292 <https://github.com/esa/pagmo2/pull/292>`__).

- **BREAKING**: unconditionally disable the CEC2013/CEC2014 problem suites on
  OSX and MinGW, as they cause build
  issues (`#266 <https://github.com/esa/pagmo2/pull/266>`__, `#292 <https://github.com/esa/pagmo2/pull/292>`__).

- **BREAKING**: the serialization backend was switched from the
  Cereal library to the Boost.serialization library. This change has
  no consequences
  for Python users, nor for C++ users who use pagmo's CMake machinery.
  For those C++ users who don't use CMake,
  this means that in order to use pagmo it is now necessary to link
  to the Boost.serialization library (`#278 <https://github.com/esa/pagmo2/pull/278>`__).

- **BREAKING**: pagmo is not any more a header-only library, it has now
  a compiled component. This change has no consequences
  for Python users, nor for C++ users who use pagmo's CMake machinery.
  For those C++ users who don't use CMake,
  this means that in order to use pagmo it is now necessary to link
  to a compiled library (`#278 <https://github.com/esa/pagmo2/pull/278>`__).

- Various performance improvements in the :cpp:class:`~pagmo::population` API (`#250 <https://github.com/esa/pagmo2/pull/250>`__).

- **BREAKING**: :class:`pygmo.problem` and :class:`pygmo.algorithm`
  cannot be used as UDPs and UDAs any more.
  This change makes the behaviour of pygmo consistent with the behaviour of pagmo (`#248 <https://github.com/esa/pagmo2/pull/248>`__).

Fix
~~~

- Fix a bug in pygmo's plotting utils (`#330 <https://github.com/esa/pagmo2/pull/330>`__).

- Fix a bug in PSO's error handling (`#323 <https://github.com/esa/pagmo2/pull/323>`__).

- Fix a bug in MOEA/D when ``m_neighbours<2`` (`#320 <https://github.com/esa/pagmo2/pull/320>`__).

- Fix type mismatches in the constrained/MO utils (`#315 <https://github.com/esa/pagmo2/pull/315>`__).

- Fix a potential deadlock when setting/getting an island's
  population/algorithm (`#309 <https://github.com/esa/pagmo2/pull/309>`__).

- Fix a build failure when pagmo is configured without Eigen3 (`#281 <https://github.com/esa/pagmo2/pull/281>`__).

- Fix a build failure in the Ipopt algorithm wrapper when using the Debian/Ubuntu Ipopt packages (`#266 <https://github.com/esa/pagmo2/pull/266>`__).

- Fix a few test suite build failures in debug mode when using recent Clang versions (`#266 <https://github.com/esa/pagmo2/pull/266>`__).

- Fix the behaviour of NSGA2 and MOEAD when the problem has equal lower/upper bounds (`#244 <https://github.com/esa/pagmo2/pull/244>`__).

- Various documentation, build system and unit testing fixes/improvements (`#243 <https://github.com/esa/pagmo2/pull/243>`__,
  `#245 <https://github.com/esa/pagmo2/pull/245>`__, `#248 <https://github.com/esa/pagmo2/pull/248>`__,
  `#257 <https://github.com/esa/pagmo2/pull/257>`__, `#262 <https://github.com/esa/pagmo2/pull/262>`__,
  `#265 <https://github.com/esa/pagmo2/pull/265>`__, `#266 <https://github.com/esa/pagmo2/pull/266>`__,
  `#279 <https://github.com/esa/pagmo2/pull/279>`__, `#287 <https://github.com/esa/pagmo2/pull/287>`__,
  `#288 <https://github.com/esa/pagmo2/pull/288>`__, `#327 <https://github.com/esa/pagmo2/pull/327>`__,
  `#328 <https://github.com/esa/pagmo2/pull/328>`__).

- The :cpp:class:`~pagmo::fork_island` UDI now properly cleans up zombie processes (`#242 <https://github.com/esa/pagmo2/pull/242>`__).

2.10 (2019-01-02)
-----------------

New
~~~

- Enable the ``py27m`` build variant for the manylinux packages (`#239 <https://github.com/esa/pagmo2/pull/239>`__).

- It is now possible to select a serialization backend other than cloudpickle. The other available
  backends are the standard :mod:`pickle` module and `dill <https://pypi.org/project/dill/>`__
  (`#229 <https://github.com/esa/pagmo2/pull/229>`__).

- The Python multiprocessing island :class:`~pygmo.mp_island` can now optionally spawn a new process for each
  evolution, rather than using a process pool (`#221 <https://github.com/esa/pagmo2/pull/221>`__).

- Python user-defined classes can now be extracted from their type-erased containers using the
  Python :class:`object` type (`#219 <https://github.com/esa/pagmo2/pull/219>`__). This allows extraction
  without knowing the exact type of the object being extracted.

Fix
~~~

- Avoid linking pygmo to the Python library on OSX with clang. This may fix the ``Fatal Python error: take_gil: NULL tstate``
  errors which are occasionally reported by users (`#230 <https://github.com/esa/pagmo2/pull/230>`__).

- Correct the detection of the Boost libraries' version in the build system (`#230 <https://github.com/esa/pagmo2/pull/230>`__).

- The Python multiprocessing island :class:`~pygmo.mp_island` should now be more robust with respect
  to serialization errors in problems/algorithms (`#229 <https://github.com/esa/pagmo2/pull/229>`__).

- Tentative fix for a pygmo build failure in Cygwin (`#221 <https://github.com/esa/pagmo2/pull/221>`__).

- Various documentation fixes and enhancements (`#217 <https://github.com/esa/pagmo2/pull/217>`__, `#218 <https://github.com/esa/pagmo2/pull/218>`__,
  `#220 <https://github.com/esa/pagmo2/pull/220>`__, `#221 <https://github.com/esa/pagmo2/pull/221>`__).

2.9 (2018-08-31)
----------------

New
~~~

- Implement the UDI extraction functionality for :cpp:class:`~pagmo::island` (`#207 <https://github.com/esa/pagmo2/pull/207>`__).

- Implement the :cpp:class:`~pagmo::fork_island` UDI (`#205 <https://github.com/esa/pagmo2/pull/205>`__).

- pip pygmo package for Python 3.7 (Linux) (`#196 <https://github.com/esa/pagmo2/pull/196>`__).

- Implement the :class:`~pygmo.decorator_problem` Python meta-problem (`#195 <https://github.com/esa/pagmo2/pull/195>`__).

- Various documentation additions (`#194 <https://github.com/esa/pagmo2/pull/194>`__).

Changes
~~~~~~~

- The build system now respects the ``CMAKE_CXX_STANDARD`` CMake setting (`#207 <https://github.com/esa/pagmo2/pull/207>`__).

- Ensure that, in :cpp:class:`~pagmo::thread_island`, the algorithm used for the evolution replaces the original algorithm
  at the end of the evolution (`#203 <https://github.com/esa/pagmo2/pull/203>`__).

- The pip pygmo package for Python 3.4 (Linux) has been dropped (`#196 <https://github.com/esa/pagmo2/pull/196>`__).

Fix
~~~

- Fix a missing ``inline`` specifier (`#206 <https://github.com/esa/pagmo2/pull/206>`__).

- Fix a bunch of missing includes in ``pagmo.hpp`` (`#202 <https://github.com/esa/pagmo2/pull/202>`__).

- Fixes for compiler warnings in GCC 8 (`#197 <https://github.com/esa/pagmo2/pull/197>`__).

- Various documentation, build system and CI fixes and enhancements (`#195 <https://github.com/esa/pagmo2/pull/195>`__,
  `#196 <https://github.com/esa/pagmo2/pull/196>`__, `#204 <https://github.com/esa/pagmo2/pull/204>`__,
  `#205 <https://github.com/esa/pagmo2/pull/205>`__, `#207 <https://github.com/esa/pagmo2/pull/207>`__).

2.8 (2018-07-12)
----------------

New
~~~

- Implement the CEC2014 problem suite (`#188 <https://github.com/esa/pagmo2/pull/188>`__, `#189 <https://github.com/esa/pagmo2/pull/189>`__).

- It is now possible to explicitly shut down the process pool of :class:`~pygmo.mp_island` (`#187 <https://github.com/esa/pagmo2/pull/187>`__).

- Start using intersphinx in the documentation (used at the moment for hyperlinking to the Python online documentation)
  (`#187 <https://github.com/esa/pagmo2/pull/187>`__).

- The constraints' tolerances for a problem can now be set via a scalar in pygmo (`#185 <https://github.com/esa/pagmo2/pull/185>`__).

Changes
~~~~~~~

- Update the copyright date to 2018 (`#190 <https://github.com/esa/pagmo2/pull/190>`__).

- **BREAKING**: user-defined islands in Python must now return the algorithm object used for the evolution in addition
  to the evolved population (`#186 <https://github.com/esa/pagmo2/pull/186>`__). This change ensures that the state of
  an algorithm executed on a pythonic island is now correctly propagated back to the original algorithm object at the end of
  an evolution.

Fix
~~~

- Fix a compilation failure involving the IHS algorithm (`#192 <https://github.com/esa/pagmo2/pull/192>`__).

- Fix a bug in the Python exposition of the DE algorithm (`#183 <https://github.com/esa/pagmo2/pull/183>`__).

- Various documentation and CI fixes and improvements (`#183 <https://github.com/esa/pagmo2/pull/183>`__,
  `#185 <https://github.com/esa/pagmo2/pull/185>`__, `#190 <https://github.com/esa/pagmo2/pull/190>`__,
  `#191 <https://github.com/esa/pagmo2/pull/191>`__).

2.7 (2018-04-13)
----------------

New
~~~

- Implement the particle swarm optimization generational (GPSO) algorithm (`#161 <https://github.com/esa/pagmo2/pull/161>`__).

- Implement the exponential natural evolution strategies (xNES) algorithm (`#142 <https://github.com/esa/pagmo2/pull/142>`__).

- Implement the improved harmony search (IHS) algorithm (`#141 <https://github.com/esa/pagmo2/pull/141>`__).

Changes
~~~~~~~

- Update pygmo's dependencies in the manylinux builds to the latest versions
  (`#144 <https://github.com/esa/pagmo2/pull/144>`__).

2.6 (2017-11-18)
----------------

Fix
~~~

- Fix a bug in NSGA2 when the bounds of the problem contain negative values (`#139 <https://github.com/esa/pagmo2/pull/139>`__).

- Various documentation fixes and improvements (`#139 <https://github.com/esa/pagmo2/pull/139>`__).

2.5 (2017-11-12)
----------------

Fix
~~~

- Fix meta-problems not forwarding the integer dimension (`#134 <https://github.com/esa/pagmo2/pull/134>`__).

- Various continuous integration fixes (`#134 <https://github.com/esa/pagmo2/pull/134>`__,
  `#136 <https://github.com/esa/pagmo2/pull/136>`__).

- Various build fixes for recent GCC versions (`#129 <https://github.com/esa/pagmo2/pull/129>`__).

- Various documentation fixes and improvements (`#121 <https://github.com/esa/pagmo2/pull/121>`__).

2.4 (2017-06-09)
----------------

New
~~~

- Initial release of the pagmo/pygmo C++ software-development kit (SDK). The purpose of the SDK is to make the process
  of writing C++ extensions for pagmo/pygmo as easy as possible. The SDK is a beta-quality feature at this time,
  and it is lightly documented - no tutorials are available yet. Please come to the `gitter channel <https://gitter.im/pagmo2/Lobby>`__
  and ask there if you are interested in it (`#110 <https://github.com/esa/pagmo2/pull/110>`__).

- Improve support for integer and mixed integer optimization (`#115 <https://github.com/esa/pagmo2/pull/115>`__).

Changes
~~~~~~~

- pygmo now depends on pagmo, and it is now effectively a separate package. That is, in order to compile and install pygmo from
  source, you will need first to install the pagmo C++ headers. Users of pip/conda are **not** affected by this change (as
  pip and conda manage dependencies automatically).

- **BREAKING**: as a consequence of the overhaul of (mixed) integer programming support in pagmo, the problem's integer part is no
  longer available as an argument when constructing algorithms such as :cpp:class:`pagmo::sga` and :cpp:class:`pagmo::nsga2`, it
  must instead be specified in the definition of the UDP via the optional ``get_nix()`` method.

Fix
~~~

- Fix a bug in the plotting submodule (`#118 <https://github.com/esa/pagmo2/pull/118>`__).

- Various documentation fixes and improvements.

2.3 (2017-05-19)
----------------

Changes
~~~~~~~

- Move from dill to cloudpickle as a serialization backend. This fixes various serialization issues reported in
  `#106 <https://github.com/esa/pagmo2/issues/106>`__.

Fix
~~~

- Various documentation fixes and improvements (`#103 <https://github.com/esa/pagmo2/pull/103>`__,
  `#104 <https://github.com/esa/pagmo2/pull/104>`__, `#107 <https://github.com/esa/pagmo2/pull/107>`__).

2.2 (2017-05-12)
----------------

New
~~~

- New tutorials (Schwefel and constrained problems) `(#91) <https://github.com/esa/pagmo2/pull/91>`__.

- Add support for `Ipopt <https://projects.coin-or.org/Ipopt>`__ `(#92) <https://github.com/esa/pagmo2/pull/92>`__.

- Implement the simple genetic algorithm (SGA) `(#93) <https://github.com/esa/pagmo2/pull/93>`__.

Changes
~~~~~~~

- Rename, fix and improve the implementation of various archipelago-related methods
  `(#94) <https://github.com/esa/pagmo2/issues/94>`__.

- Remove the use of atomic counters in the problem `(#79) <https://github.com/esa/pagmo2/issues/79>`__.

Fix
~~~

- Various documentation fixes/improvements, headers sanitization, etc.
