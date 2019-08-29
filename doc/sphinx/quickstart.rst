.. _quickstart:

Quick start 
============

.. _getting_started_c++:

C++
---

After following the :ref:`installation guide <install>`,
you will be able to compile and run your first C++ pagmo program:

.. literalinclude:: ../../tutorials/getting_started.cpp
   :language: c++
   :linenos:

Place it into a ``getting_started.cpp`` text file and compile it
(for example) with:

.. code-block:: console

   $ g++ -O2 -DNDEBUG -std=c++11 getting_started.cpp -pthread -lpagmo -lboost_serialization -ltbb

If you installed pagmo in a non-standard path, such as the ``.local`` directory
in your ``$HOME`` on a Unix installation (e.g., ``/home/username/.local``),
the compiler will need assistance to locate the pagmo headers and libraries.
E.g., you may need a command such as:

.. code-block:: console

   $ g++ -O2 -DNDEBUG -std=c++11 getting_started.cpp -pthread -lpagmo -lboost_serialization -ltbb -I /home/username/.local/include -L /home/username/.local/lib -Wl,-R/home/username/.local/lib

If you installed pagmo with support for optional 3rd party libraries,
you might need to add additional switches to the command-line invocation
of the compiler. For instance, if you enabled the optional NLopt support,
you will have to link your executable to the
``nlopt`` library:

.. code-block:: console

   $ g++ -O2 -DNDEBUG -std=c++11 getting_started.cpp -pthread -lpagmo -lboost_serialization -ltbb -lnlopt

We recommend to use pagmo's CMake support in order to simplify
the build process of code depending on pagmo (see next section).

Using pagmo with CMake
^^^^^^^^^^^^^^^^^^^^^^

As a part of the pagmo installation, a group of CMake files is installed into
``CMAKE_INSTALL_PREFIX/lib/cmake/pagmo``.
This bundle, which is known in the CMake lingo as a
`config-file package <https://cmake.org/cmake/help/v3.3/manual/cmake-packages.7.html>`__,
facilitates the detection and use of pagmo from other CMake-based projects.
pagmo's config-file package, once loaded, provides
an imported target called ``Pagmo::pagmo`` which encapsulates all the information
necessary to use pagmo. That is, linking to
``Pagmo::pagmo`` ensures that pagmo's include directories are added to
the include
path of the compiler, and that the libraries
on which pagmo depends are brought into the link chain.

For instance, a ``CMakeLists.txt`` file for the simple ``getting_started.cpp``
program presented earlier may look like this:

.. code-block:: cmake

   # pagmo needs at least CMake 3.2.
   cmake_minimum_required(VERSION 3.2.0)

   # The name of our project.
   project(sample_project)

   # Look for an installation of pagmo in the system.
   find_package(Pagmo REQUIRED)

   # Create an executable, and link it to the Pagmo::pagmo imported target.
   # This ensures that, in the compilation of 'getting_started', pagmo's include
   # dirs are added to the include path of the compiler and that pagmo's
   # dependencies are transitively linked to 'getting_started'.
   add_executable(getting_started getting_started.cpp)
   target_link_libraries(getting_started Pagmo::pagmo)

   # This line indicates to your compiler
   # that C++11 is needed for the compilation.
   # Not strictly necessary with a recent-enough compiler.
   set_property(TARGET getting_started PROPERTY CXX_STANDARD 11)

.. _getting_started_py:

Python
------

If you have successfully installed pygmo following the
:ref:`installation guide <install>`, you can try the following script:

.. literalinclude:: docs/examples/getting_started.py
   :language: python
   :linenos:

Place it into a ``getting_started.py`` text file and run it with:

.. code-block:: console

   $ python getting_started.py

We recommend the use of `Jupyter <https://jupyter.org/>`__ or
`IPython <https://ipython.org/>`__ to enjoy pygmo the most.

