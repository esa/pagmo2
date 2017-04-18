.. _install:

Installation guide
==================

.. contents::


C++
---

Pagmo is a header-only library which has the following third party dependencies:

* `Boost <http://www.boost.org/>`_, **mandatory**, header-only (needs the libraries only if you
  intend to compile the python bindings)
* `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_, optional, header-only
* `NLopt <http://ab-initio.mit.edu/wiki/index.php/NLopt>`_, optional, requires linking

After making sure the dependencies above are installed in your system, you can download the
pagmo source code from the `GitHub release page <https://github.com/esa/pagmo2/releases>`_. Alternatively,
and if you like living on the bleeding edge, you can get the very latest version of pagmo via the ``git``
command:

.. code-block:: bash

   git clone https://github.com/esa/pagmo2.git

Once you are in pagmo's source tree, you must configure your build using ``cmake``. This will allow
you to enable support for optional dependencies, configure the install destination, etc. When done,
you can install pagmo via the command

.. code-block:: bash

   make install

The headers will be installed in the ``CMAKE_INSTALL_PREFIX/include`` directory. To check that all went well
compile the :ref:`quick-start example <getting_started_c++>`.

-----------------------------------------------------------------------

Python
------
The python module corresponding to pagmo is called pygmo. There are various options for the installation
of pygmo:

* `conda <https://conda.io/docs/>`_,
* `pip <https://pip.pypa.io/en/stable/>`_,
* installation from source.

The following table summarizes the pros and cons of the various installation methods:

========= ============ ============ ========== ========== ================ ==========
Method    Linux Py 2.7 Linux Py 3.x OSX Py 2.7 OSX Py 3.x Win Py 2.7       Win Py 3.x
========= ============ ============ ========== ========== ================ ==========
conda     64bit        64bit        64bit      64bit      ✘                64bit
pip       64bit        64bit        ✘          ✘          64bit (MinGW)    64bit (MinGW)
source    32/64bit     32/64bit     32/64bit   32/64bit   32/64bit (MinGW) 32/64bit
========= ============ ============ ========== ========== ================ ==========

In general, we recommend the use of `conda <https://conda.io/docs/>`_: in addition to making the installation
of pygmo easy, it also provides user-friendly access to a wealth of packages from the scientific python
ecosystem. Conda is a good default choice in Linux and OSX.

In Windows, the situation is a bit more complicated. The first issue is that the compiler used by conda
for Python 2.7 is too old to compile pygmo, and thus we cannot provide conda packages for Python 2.7
(however, we do provide conda packages for Python 3.x). The second issue is that the Windows platform
lacks a free Fortran compiler that can interoperate with Visual C++ (the compiler used by conda on Windows).
Thus, the pygmo packages for conda on Windows might lack some Fortran-based features available on Linux and OSX
(e.g., the wrapper for the Ipopt solver).

Thus, in order to provide a better experience to our Windows users, we publish `pip <https://pip.pypa.io/en/stable/>`_
packages for pygmo built with `MinGW <https://mingw-w64.org/doku.php>`_. These packages allow us both to support Python 2.7
and to provide a full-featured pygmo on Windows, thanks to the ``gfortran`` compiler. The pip packages are also available on
Linux for those users who might prefer pip to conda, but they are **not** available on OSX.

Finally, it is always possible to compile pygmo from source. This is the most flexible and powerful option, but of course
also the least user-friendly.

.. note::
   As a general policy, we are committed to providing packages for Python 2.7 and for the latest two versions of Python 3.x.


Installation with conda
^^^^^^^^^^^^^^^^^^^^^^^
The installation of pygmo with conda is straightforward. We just need to add ``conda-forge`` to the channels,
and then we can immediately install pygmo:

.. code-block:: bash

   conda config --add channels conda-forge 
   conda install pygmo

Please refer to the `conda documentation <https://conda.io/docs/>`_ for instructions on how to setup and manage
your conda installation.


Installation with pip
^^^^^^^^^^^^^^^^^^^^^
The installation of pygmo with conda is also straightforward:

.. code-block:: bash

   pip install pygmo

If you want to install pygmo for a single user instead of system-wide, which is in general a good idea, you can do:

.. code-block:: bash

   pip install --user pygmo


Installation from source
^^^^^^^^^^^^^^^^^^^^^^^^
To build the module from source you need to have the Boost.Python libraries installed and to activate the cmake
``PAGMO_BUILD_PYGMO`` option.

Check carefully what python version is detected and what libraries are linked to. In particular, select the correct Boost.Python
version according to the python version (2 or 3) you want to compile the module for.

The ``CMAKE_INSTALL_PREFIX`` will be used to construct the final location of headers and Python module after install.

When done, type (in your build directory):

.. code-block:: bash

   make install

To check that all went well fire-up your python console and try the example in :ref:`quick-start example <getting_started_py>`.
