Changelog
=========

2.4 (unreleased)
----------------

New
~~~

- Improve support for integer and mixed integer optimization (`#115 <https://github.com/esa/pagmo2/pull/115>`_).

Fix
~~~

- Fix a bug in the plotting submodule (`#118 <https://github.com/esa/pagmo2/pull/118>`_).

- Various documentation fixes and improvements.

2.3 (2017-05-19)
----------------

Changes
~~~~~~~

- Move from dill to cloudpickle as a serialization backend. This fixes various serialization issues reported in
  `#106 <https://github.com/esa/pagmo2/issues/106>`_.

Fix
~~~

- Various documentation fixes and improvements (`#103 <https://github.com/esa/pagmo2/issues/103>`_,
  `#104 <https://github.com/esa/pagmo2/issues/104>`_, `#107 <https://github.com/esa/pagmo2/issues/107>`_).

2.2 (2017-05-12)
----------------

New
~~~

- New tutorials (Schwefel and constrained problems) `(#91) <https://github.com/esa/pagmo2/pull/91>`_.

- Add support for `Ipopt <https://projects.coin-or.org/Ipopt>`_ `(#92) <https://github.com/esa/pagmo2/pull/92>`_.

- Implement the simple genetic algorithm (SGA) `(#93) <https://github.com/esa/pagmo2/pull/93>`_.

Changes
~~~~~~~

- Rename, fix and improve the implementation of various archipelago-related methods
  `(#94) <https://github.com/esa/pagmo2/issues/94>`_.

- Remove the use of atomic counters in the problem `(#79) <https://github.com/esa/pagmo2/issues/79>`_.

Fix
~~~

- Various documentation fixes/improvements, headers sanitization, etc.
