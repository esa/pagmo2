.. _install:

Installation
============

Dependencies
------------

pagmo is written in modern C++, and it requires a compiler able to understand
at least C++17. pagmo is known to run on the following setups:

* GCC 7 and later versions on GNU/Linux,
* Clang 4 and later versions on GNU/Linux,
* MSVC 2015 and later versions on Windows,
* Clang 4 and later versions on Windows
  (with the ``clang-cl`` driver for MSVC),
* MinGW GCC 8 on Windows,
* Clang on OSX (Xcode 6.4 and later),
* Clang on FreeBSD.

The pagmo C++ library has the following **mandatory** dependencies:

* the `Boost <https://www.boost.org/>`__ C++ libraries (at least version 1.60),
* the `Intel TBB <https://www.threadingbuildingblocks.org/>`__ library.

Additionally, pagmo has the following **optional** dependencies:

* `Eigen3 <http://eigen.tuxfamily.org/index.php?title=Main_Page>`__ (which is required
  by some algorithms, e.g., :cpp:class:`pagmo::cmaes`), version 3.3 or later,
* `NLopt <https://nlopt.readthedocs.io/en/latest/>`__ (which is required by
  the :cpp:class:`pagmo::nlopt` wrapper), version 2.6 or later,
* `Ipopt <https://projects.coin-or.org/Ipopt>`__ (which is required by
  the :cpp:class:`pagmo::ipopt` wrapper).

Additionally, `CMake <https://cmake.org/>`__ is the build system used by
pagmo and it must also be available when
installing from source (the minimum required version is 3.3).

Packages
--------

pagmo packages are available from a variety
of package managers on several platforms.

Conda
^^^^^

pagmo is available via the `conda <https://conda.io/docs/>`__ package manager for Linux, OSX and Windows
thanks to the infrastructure provided by `conda-forge <https://conda-forge.org/>`__.
Two packages are available:

* `pagmo <https://anaconda.org/conda-forge/pagmo>`__, which contains the pagmo shared library,
* `pagmo-devel <https://anaconda.org/conda-forge/pagmo-devel>`__,
  which contains the pagmo headers and the
  CMake support files.

In order to install pagmo via conda, you just need to add ``conda-forge``
to the channels, and then we can immediately install pagmo:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict
   $ conda install pagmo pagmo-devel

The conda packages for pagmo are maintained by the core development team,
and they are regularly updated when new pagmo versions are released.

Please refer to the `conda documentation <https://conda.io/docs/>`__ for instructions on how to setup and manage
your conda installation.

Arch Linux
^^^^^^^^^^

pagmo is also available on the `Arch User Repository
<https://aur.archlinux.org>`__ (AUR) in Arch Linux. It is
recommended to use an AUR helper like
`yay <https://aur.archlinux.org/packages/yay/>`__ or
`pikaur <https://aur.archlinux.org/packages/pikaur/>`__ for ease of installation.
See the `AUR helpers <https://wiki.archlinux.org/index.php/AUR_helpers>`__ page on
the Arch Linux wiki for more info.

.. note::

   To install pagmo with optional dependency support like nlopt or ipopt,
   make sure to install the optional dependencies before installing the pagmo
   package.

Install optional dependencies:

.. code-block:: console

    $ yay -S coin-or-ipopt eigen nlopt

Install pagmo:

.. code-block:: console

    $ yay -S pagmo

FreeBSD
^^^^^^^

A FreeBSD port via `pkg
<https://www.freebsd.org/doc/handbook/pkgng-intro.html>`__ has been created for
pagmo. In order to install pagmo using pkg, execute the following command:

.. code-block:: console

   $ pkg install pagmo2

Homebrew
^^^^^^^^

A `Homebrew <https://brew.sh/>`__ recipe for pagmo is also available. In order to install
pagmo on OSX with Homebrew, it is sufficient to execute the following command:

.. code-block:: console

   $ brew install pagmo


Installation from source
------------------------

After making sure the dependencies are installed on your system, you can
download the pagmo source code from the
`GitHub release page <https://github.com/esa/pagmo2/releases>`__. Alternatively,
and if you like living on the bleeding edge, you can get the very latest
version of pagmo via ``git``:

.. code-block:: console

   $ git clone https://github.com/esa/pagmo2.git

We follow the usual PR-based development workflow, thus pagmo's ``master``
branch is normally kept in a working state.

After downloading and/or unpacking pagmo's
source code, go to pagmo's
source tree, create a ``build`` directory and ``cd`` into it. E.g.,
on a Unix-like system:

.. code-block:: console

   $ cd /path/to/pagmo
   $ mkdir build
   $ cd build

Once you are in the ``build`` directory, you must configure your build
using ``cmake``. This will allow you to enable support for optional
dependencies, configure the install destination, etc.

The following options are currently recognised by pagmo’s build system:

* ``PAGMO_BUILD_TESTS``: build the test suite (defaults to ``OFF``),
* ``PAGMO_BUILD_TUTORIALS``: build the C++
  :ref:`tutorials <tutorial>` (defaults to ``OFF``),
* ``PAGMO_WITH_EIGEN3``: enable features depending on `Eigen3 <http://eigen.tuxfamily.org/index.php?title=Main_Page>`__
  (defaults to ``OFF``),
* ``PAGMO_WITH_NLOPT``: enable the `NLopt <https://nlopt.readthedocs.io/en/latest/>`__
  wrappers (defaults to ``OFF``),
* ``PAGMO_WITH_IPOPT``: enable the `Ipopt <https://projects.coin-or.org/Ipopt>`__
  wrapper (defaults to ``OFF``).

Additionally, there are various useful CMake variables you can set, such as:

* ``CMAKE_BUILD_TYPE``: the build type (``Release``, ``Debug``, etc.),
  defaults to ``Release``.
* ``CMAKE_INSTALL_PREFIX``: the path into which pagmo will be installed
  (e.g., this defaults to ``/usr/local`` on Unix-like platforms).
* ``CMAKE_PREFIX_PATH``: additional paths that will be searched by CMake
  when looking for dependencies.

Please consult `CMake's documentation <https://cmake.org/cmake/help/latest/>`_
for more details about CMake's variables and options.

A typical CMake invocation for pagmo may look something like this:

.. code-block:: console

   $ cmake ../ -DPAGMO_BUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=~/.local

That is, we build the test suite and we
will be installing pagmo into our home directory into the ``.local``
subdirectory. If CMake runs without errors, we can then proceed to actually
building pagmo:

.. code-block:: console

   $ cmake --build .

This command will build the pagmo library and, if requested, the test suite.
Next, we can install pagmo with the command:

.. code-block:: console

   $ cmake  --build . --target install

This command will install the pagmo library and header files to
the directory tree indicated by the ``CMAKE_INSTALL_PREFIX`` variable.

If enabled, the test suite can be executed with the command:

.. code-block:: console

   $ cmake  --build . --target test

.. note::

   On Windows, in order to execute the test suite you have to ensure that the
   ``PATH`` variable includes the directory that contains the pagmo
   DLL (otherwise the tests will fail to run).

Getting help
------------

If you run into troubles installing pagmo, please do not hesitate
to contact us either through our `gitter channel <https://gitter.im/pagmo2/Lobby>`__
or by opening an issue report on `github <https://github.com/esa/pagmo2/issues>`__.
