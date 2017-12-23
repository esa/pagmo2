Changelog
=========

2.7 (unreleased)
----------------

New
~~~

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
