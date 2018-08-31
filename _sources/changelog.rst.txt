Changelog
=========

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
