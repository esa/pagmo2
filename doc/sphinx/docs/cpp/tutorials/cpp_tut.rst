.. _cpp_tutorials:

C++ tutorials
================

Basics
^^^^^^

After following the :ref:`install` you will be able to compile and run C++ pagmo
programs. The following example shows the use with no multithreading of an algoritmic evolution:

.. literalinclude:: ../../../docs/examples/nsga2.cpp
   :language: c++
   :linenos:

Place it into a ``nsg2.cpp`` text file and compile it (for example) with:

.. code-block:: bash

   g++ -O2 -DNDEBUG -std=c++11 getting_started.cpp

  
