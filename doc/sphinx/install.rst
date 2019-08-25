.. _install:

Installation guide
==================

.. _cpp_install:

C++
---

Requirements
^^^^^^^^^^^^

The pagmo C++ library has the following **mandatory** dependencies:

* the `Boost <https://www.boost.org/>`__ C++ libraries,
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

* Python TODO,
* the pagmo C++ library,
* the `Boost.Python <https://www.boost.org/>`__ library, TODO
* `NumPy <http://www.numpy.org/>`__, the standard Python array library
* `cloudpickle <https://github.com/cloudpipe/cloudpickle>`__, a
  package that extends Python's serialization
  capabilities.

Additionally, pygmo has the following **optional** dependencies:

* dill TODO,
* matplotlib TODO.

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
similar to the instrcution for a pagmo installation TODO link:

* install the required dependencies (including the pagmo
  C++ library),
* download/checkout the source code,
* use CMake to configure and build pygmo.

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
:ref:`quick-start example <getting_started_py>`.

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

Pip
"""

There are various options for the installation of pygmo:

* `conda <https://conda.io/docs/>`__
* `pip <https://pip.pypa.io/en/stable/>`__
* installation from source.

The following table summarizes the pros and cons of the various installation methods:

========= ============ ============ ========== ========== ============== ==============
Method    Linux Py 2.7 Linux Py 3.x OSX Py 2.7 OSX Py 3.x Win Py 2.7     Win Py 3.x
========= ============ ============ ========== ========== ============== ==============
*conda*   ✔             ✔            ✔         ✔           ✘             ✔ 
*pip*     ✔             ✔            ✘         ✘           ✔ (MinGW)     ✔ (MinGW)
*source*  ✔             ✔            ✔         ✔           ✔ (MinGW)     ✔ 
========= ============ ============ ========== ========== ============== ==============

In general, we recommend the use of `conda <https://conda.io/docs/>`__: in addition to making the installation
of pygmo easy, it also provides user-friendly access to a wealth of packages from the scientific Python
ecosystem. Conda is a good default choice in Linux and OSX.

In Windows, the situation is a bit more complicated. The first issue is that the compiler used by conda
for Python 2.7 is too old to compile pygmo, and thus we cannot provide conda packages for Python 2.7
(however, we do provide conda packages for Python 3.x). The second issue is that the Windows platform
lacks a free Fortran compiler that can interoperate with Visual C++ (the compiler used by conda on Windows).
Thus, the pygmo packages for conda on Windows might lack some Fortran-based features available on Linux and OSX
(e.g., the wrapper for the Ipopt solver).

Thus, in order to provide a better experience to our Windows users, we publish `pip <https://pip.pypa.io/en/stable/>`__
packages for pygmo built with `MinGW <http://mingw-w64.org/doku.php>`__. These packages allow us both to support Python 2.7
and to provide a full-featured pygmo on Windows, thanks to the ``gfortran`` compiler. The pip packages are also available on
Linux for those users who might prefer pip to conda, but they are **not** available on OSX.

Finally, it is always possible to compile pygmo from source. This is the most flexible and powerful option, but of course
also the least user-friendly.

.. note::

   As a general policy, we are committed to providing packages for Python 2.7 and for the latest two versions of Python 3.x.

.. note::

   All the binary packages are compiled in 64-bit mode.


Installation with conda
^^^^^^^^^^^^^^^^^^^^^^^
The installation of pygmo with conda is straightforward. We just need to add ``conda-forge`` to the channels,
and then we can immediately install pygmo:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda install pygmo

conda will automatically install all of pygmo's dependencies for you. Please refer to the `conda documentation <https://conda.io/docs/>`__
for instructions on how to setup and manage your conda installation.


Installation with pip
^^^^^^^^^^^^^^^^^^^^^
The installation of pygmo with pip is also straightforward:

.. code-block:: console

   $ pip install pygmo

Like conda, also pip will automatically install all of pygmo's dependencies for you. If you want to install pygmo for a single user instead of
system-wide, which is in general a good idea, you can do:

.. code-block:: console

   $ pip install --user pygmo


Installation from source
^^^^^^^^^^^^^^^^^^^^^^^^
For an installation from source, pygmo has the following dependencies:

* pagmo (i.e., the C++ headers of the pagmo library need to be installed before attempting
  to compile pygmo),
* `Boost.Python <https://www.boost.org/doc/libs/1_63_0/libs/python/doc/html/index.html>`__
* `NumPy <http://www.numpy.org/>`__ (note that NumPy's development headers must be installed as well).

Note that, at the present time, the versions of pygmo and pagmo must be exactly identical for the compilation of pygmo
to be successful, otherwise the build process will error out. If you are updating pagmo/pygmo to a later version,
make sure to install the new pagmo version before compiling the new pygmo version.

To build the module from source you need to **activate** the ``PAGMO_BUILD_PYGMO`` cmake option and **deactivate** the ``PAGMO_BUILD_PAGMO`` option.
Check carefully what Python version and what libraries/include paths are detected (in particular, on systems with multiple Python versions
it can happen that CMake detects the headers from a Python version and the Python library from another version).
The ``CMAKE_INSTALL_PREFIX`` variable will be used to construct the final location of headers and Python module after install.

When done, type (in your build directory):

.. code-block:: console

   $ make install

To check that all went well fire-up your Python console and try the example in :ref:`quick-start example <getting_started_py>`.

Installation on Arch Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
