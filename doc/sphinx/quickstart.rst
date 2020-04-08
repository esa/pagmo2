Quick start
===========

After following the :ref:`installation guide <install>`,
you will be able to compile and run your first C++ pagmo program:

.. literalinclude:: ../../tutorials/getting_started.cpp
   :caption: getting_started.cpp
   :language: c++
   :linenos:

Place it into a ``getting_started.cpp`` text file and compile it
(for example) with:

.. code-block:: console

   $ g++ -O2 -DNDEBUG -std=c++17 getting_started.cpp -pthread -lpagmo -lboost_serialization -ltbb

If you installed pagmo in a non-standard path, such as the ``.local`` directory
in your ``$HOME`` on a Unix installation (e.g., ``/home/username/.local``),
the compiler will need assistance to locate the pagmo headers and libraries.
E.g., you may need a command such as:

.. code-block:: console

   $ g++ -O2 -DNDEBUG -std=c++17 getting_started.cpp -pthread -lpagmo -lboost_serialization -ltbb -I /home/username/.local/include -L /home/username/.local/lib -Wl,-R/home/username/.local/lib

If you installed pagmo with support for optional 3rd party libraries,
you might need to add additional switches to the command-line invocation
of the compiler.

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

   # pagmo needs at least CMake 3.3.
   cmake_minimum_required(VERSION 3.3)

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
   # that C++17 is needed for the compilation.
   set_property(TARGET getting_started PROPERTY CXX_STANDARD 17)

Place this ``CMakeLists.txt`` and the ``getting_started.cpp`` files
in the same directory, and create a ``build`` subdirectory. From
the ``build`` subdirectory, execute these commands to produce
the ``getting_started`` executable:

.. code-block:: console

   $ cmake ../ -DCMAKE_BUILD_TYPE=Release
   $ cmake  --build .

Please refer to the `CMake documentation <https://cmake.org/documentation/>`__
for more information on how to use CMake.
