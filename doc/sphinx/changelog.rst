Changelog
=========

2.5 (unreleased)
----------------

Fix
~~~

- Various documentation fixes and improvements.

2.4 (2017-06-09)
----------------

New
~~~

- Initial release of the pagmo/pygmo C++ software-development kit (SDK). The purpose of the SDK is to make the process
  of writing C++ extensions for pagmo/pygmo as easy as possible. The SDK is a beta-quality feature at this time,
  and it is lightly documented - no tutorials are available yet. Please come to the gitter channel and ask there if you are interested
  in it (`#110 <https://github.com/esa/pagmo2/pull/110>`__).

- Improve support for integer and mixed integer optimization (`#115 <https://github.com/esa/pagmo2/pull/115>`__).

Changes
~~~~~~~

- pygmo now depends on pagmo, and it is now effectively a separate package. That is, in order to compile and install pygmo from
  source, you will need first to install the pagmo C++ headers. Users of pip/conda are **not** affected by this change.

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

- Various documentation fixes and improvements (`#103 <https://github.com/esa/pagmo2/issues/103>`__,
  `#104 <https://github.com/esa/pagmo2/issues/104>`__, `#107 <https://github.com/esa/pagmo2/issues/107>`__).

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
