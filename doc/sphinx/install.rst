.. _install:

Installation guide
==================

.. contents::


C++
---

Pagmo is a header-only library which has the following third party dependencies:

* `Boost <http://www.boost.org/>`_, headers only (needs the libraries only if you intend to compile the python bindings)
* `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_, headers only (optional)


After making sure the dependencies above are installed in your system and their headers visible to your compiler, you can download 
pagmo source via the ``git`` command 

.. code-block:: bash

   git clone https://github.com/esa/pagmo2.git

and configure your build using ``cmake``. When done, type (in your build directory):

.. code-block:: bash

   make install

The headers will be installed in the CMAKE_INSTALL_PREFIX/include directory. To check that all went well compile the :ref:`quick-start example <getting_started_c++>`.

-----------------------------------------------------------------------

Python
------
The python module correponding to pagmo is called pygmo
It can be installed either directly from ``conda`` or ``pip`` or by building the module from source.

Installing with pip
^^^^^^^^^^^^^^^^^^^
The Python package pygmo (Python binding of the C++ code) can be installed using ``pip`` or ``conda``:

.. code-block:: bash

   pip install pygmo

or

.. code-block:: bash

   conda install pygmo

Building the module
^^^^^^^^^^^^^^^^^^^

To build the module you need to have the Boost Python libraries installed and to activate the BUILD_PYGMO option from within ``cmake``.

Check carefully what Python version is detected and what libraries are linked to. In particular select the correct boost_python
according to the Python version (2 or 3) you want to compile the module for.

The CMAKE_INSTALL_PREFIX will be used to construct the final location of headers and Python module after install.

When done, type (in your build directory):

.. code-block:: bash

   make install

To check that all went well fire-up your Python console and try the example in :ref:`quick-start example <getting_started_py>`.
