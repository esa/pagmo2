.. _install:

Installation guide
==================

.. _cpp_install:

C++
---

Requirements
^^^^^^^^^^^^

The pagmo C++ library has the following **mandatory** dependencies:

* the `Boost <https://www.boost.org/>`__ C++ libraries (at least version 1.60),
* the `Intel TBB <https://www.threadingbuildingblocks.org/>`__ library.

Additionally, pagmo has the following **optional** dependencies:

* `Eigen3 <http://eigen.tuxfamily.org/index.php?title=Main_Page>`__ (which is required
  by some algorithms, e.g., :cpp:class:`pagmo::cmaes`),
* `NLopt <https://nlopt.readthedocs.io/en/latest/>`__ (which is required by
  the :cpp:class:`pagmo::nlopt` wrapper),
* `Ipopt <https://projects.coin-or.org/Ipopt>`__ (which is required by
  the :cpp:class:`pagmo::ipopt` wrapper).

Installation from source
^^^^^^^^^^^^^^^^^^^^^^^^

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

To check that all went well, compile the
:ref:`quick-start example <getting_started_c++>`.

Packages
^^^^^^^^

pagmo is also available from a variety of package managers on
various platforms.

Conda
"""""

pagmo is available via the `conda <https://conda.io/docs/>`__ package manager for Linux, OSX and Windows
thanks to the infrastructure provided by `conda-forge <https://conda-forge.org/>`__.
In order to install pagmo via conda, you just need to add ``conda-forge``
to the channels, and then we can immediately install pagmo:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda install pagmo

The conda packages for pagmo are maintained by the core development team,
and they are regularly updated when new pagmo versions are released.

Please refer to the `conda documentation <https://conda.io/docs/>`__ for instructions on how to setup and manage
your conda installation.

Arch Linux
""""""""""

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
"""""""

A FreeBSD port via `pkg
<https://www.freebsd.org/doc/handbook/pkgng-intro.html>`__ has been created for
pagmo. In order to install pagmo using pkg, execute the following command:

.. code-block:: console

   $ pkg install pagmo2

Homebrew
""""""""

A `Homebrew <https://brew.sh/>`__ recipe for pagmo is also available. In order to install
pagmo on OSX with Homebrew, it is sufficient to execute the following command:

.. code-block:: console

   $ brew install pagmo

.. _py_install:

Python
------

Requirements
^^^^^^^^^^^^

The Python module corresponding to pagmo is called pygmo.
pygmo has the following **mandatory** dependencies:

* `Python <https://www.python.org/>`__,
* the pagmo C++ library,
* the `Boost.Python <https://github.com/boostorg/python>`__ library,
* `NumPy <http://www.numpy.org/>`__, the standard Python array library
* `cloudpickle <https://github.com/cloudpipe/cloudpickle>`__, a
  package that extends Python's serialization
  capabilities.

Additionally, pygmo has the following **optional** dependencies:

* `dill <https://dill.readthedocs.io>`__, which can be used as an
  alternative serialization backend,
* `Matplotlib <https://matplotlib.org/>`__, which is used by a few
  plotting utilities.

The presence of dill and/or matplotlib will be detected at runtime
by pygmo, thus they need not to be present when installing/compiling
pygmo.

.. note::

   Currently, pygmo must always be installed and upgraded in lockstep
   with pagmo. That is, the versions of pagmo and pygmo must match
   *exactly*, and if you want to upgrade pagmo, you will have to upgrade
   pygmo as well to the exact same version. In the future we may
   relax this requirement.

Installation from source
^^^^^^^^^^^^^^^^^^^^^^^^

Currently, pagmo and pygmo reside in the same source tree. Thus,
the instructions to install pygmo from source are largely
similar to the instrcution for a :ref:`pagmo installation <cpp_install>`:

* install the required dependencies (including the pagmo
  C++ library),
* download/checkout the source code,
* use CMake to configure, build and install pygmo.

In order to build pygmo, you will have to **disable** the
``PAGMO_BUILD_PAGMO`` option (which is ``ON`` by default)
and **enable** the ``PAGMO_BUILD_PYGMO`` option (which is
``OFF`` by default). There are no other pygmo-specific
CMake options to set. pygmo will detect automatically from the
pagmo C++ installation in use which optional features
were enabled (e.g., Eigen3, Ipopt, etc.).

A critical setting for a pygmo installation is the
value of the ``CMAKE_INSTALL_PREFIX`` variable. The pygmo
build system will attempt to construct an appropriate
installation path for the Python module by combining
the value of ``CMAKE_INSTALL_PREFIX`` with the directory
paths of the Python installation in use in a platform-dependent
manner.

For instance, on a typical Linux installation
of Python 3.6,
``CMAKE_INSTALL_PREFIX`` will be set by default to
``/usr/local``, and the pygmo build system will
append ``lib/python3.6/site-packages`` to the install prefix.
Thus, the overall install path for the pygmo module will be
``/usr/local/lib/python3.6/site-packages``. If you want
to avoid system-wide installations (which require root
privileges), on Unix-like system it is possible to set
the ``CMAKE_INSTALL_PREFIX`` variable to the directory
``.local`` in your ``$HOME`` (e.g., ``/home/username/.local``).
The pygmo install path will then be, in this case,
``/home/username/.local/lib/python3.6/site-packages``,
a path which is normally recognised by Python installations
without the need to modify the ``PYTHONPATH`` variable.
If you install pygmo in non-standard prefixes, you may
have to tweak your Python installation in order for the
Python interpreter to find the pygmo module.

To check that all went well, try running the
:ref:`quick-start example <getting_started_py>`. You can also
test the pygmo installation by running the test suite:

.. code-block:: python

   >>> import pygmo
   >>> pygmo.test.run_test_suite(1)

Packages
^^^^^^^^

pygmo is also available from a variety of package managers on
various platforms.

Conda
"""""

pygmo is available via the `conda <https://conda.io/docs/>`__ package manager for Linux, OSX and Windows
thanks to the infrastructure provided by `conda-forge <https://conda-forge.org/>`__.
In order to install pygmo via conda, you just need to add ``conda-forge``
to the channels, and then we can immediately install pygmo:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda install pygmo

The conda packages for pygmo are maintained by the core development team,
and they are regularly updated when new pygmo versions are released.
Note however that, due to various technical issues, the Python 2.7
conda packages for Windows are **not** available. If you need pygmo
on Windows on a Python 2.7 installation, the pip packages can be used
(see below).

Please refer to the `conda documentation <https://conda.io/docs/>`__ for instructions on how to setup and manage
your conda installation.

pip
"""

pygmo is also available via the `pip <https://pip.pypa.io/en/stable/>`__ package
installer. The installation of pygmo with pip is straightforward:

.. code-block:: console

   $ pip install pygmo

Like conda, also pip will automatically install all of pygmo's
dependencies for you.
If you want to install pygmo for a single user instead of
system-wide, which is in general a good idea, you can do:

.. code-block:: console

   $ pip install --user pygmo

An even better idea is to make use of Python's
`virtual environments <https://virtualenv.pypa.io/en/latest/>`__.

The pip packages for pygmo are maintained by the core development team,
and they are regularly updated when new pygmo versions are released.
We provide pip packages for Linux and Windows (both Python 2.7 and 3.x),
but **not** for OSX.

Arch Linux
""""""""""

Pygmo is available on the `Arch User Repository
<https://aur.archlinux.org>`__ (AUR) in Arch Linux. It is
recommended to use an AUR helper like
`yay <https://aur.archlinux.org/packages/yay/>`__ or
`pikaur <https://aur.archlinux.org/packages/pikaur/>`__ for ease of installation.
See the `AUR helpers <https://wiki.archlinux.org/index.php/AUR_helpers>`__ page on
the Arch Linux wiki for more info.

Install ``python-pygmo``:

.. code-block:: console

   $ yay -S python-pygmo
