Preliminaries
=============

Organization of the library
---------------------------

pagmo consists of a set of header files and a single
compiled library. The headers are collected hierarchically
in the ``pagmo/`` subdirectory. There is a global
``pagmo/pagmo.hpp`` header which includes
the public API in its entirety. In order to reduce
compilation times, however, we recommend
to include only the header files which are actually
needed in your code (e.g., ``pagmo/problem.hpp``,
``pagmo/algorithms/de.hpp``, etc.).

.. warning::

   Do **not** include headers from the ``pagmo/detail`` subdirectory! They contain
   implementation details which may change from version to version in incompatible ways.

Namespaces
----------

All of pagmo's public classes and functions are located
directly in the ``pagmo::`` namespace. There are no other
namespaces in pagmo's public API.

API and ABI stability
---------------------

Currently, pagmo guarantees API and ABI stability across
patch versions. That is, version x.y.n of the pagmo library
is both API and ABI compatible with version x.y.m.

While the binary interface is very likely to change across
minor versions (e.g., from 2.11 to 2.12), incompatible
API changes between minor versions are less frequent,
and they are always explicitly
documented in the :ref:`changelog <changelog>` with
the **BREAKING** tag.

.. _cpp_tut_type_erasure:

Type erasure
------------

The current incarnation of pagmo eschews traditional
object-oriented programming (OOP) techniques in favour of a
more modern approach based on `type erasure <https://en.wikipedia.org/wiki/Type_erasure>`__.
What does that mean exactly?

A widely-used approach in C++ optimisation libraries
is to leverage the language's OOP facilities in order
to allow the users to define their own optimisation
problems [#coptlib]_. In practical terms, this means that
the definition of an optimisation problem requires
the user to write a new class which derives from a "base"
optimisation problem class (e.g., see the ``TNLP``
class from the `Ipopt <https://github.com/coin-or/Ipopt>`__
optimisation library). Pure virtual methods from the
base class need then to be implemented in the derived class
in order to provide the implementation of the objective
function, of its gradient, etc.

Although perfectly valid, the OOP approach has a couple
of serious drawbacks:

* it introduces a tight coupling between the user's
  code and the optimisation library's code,
* due to the way traditional OOP is implemented in C++,
  it forces the use of reference semantics
  (i.e., you have to deal with "pointers to base
  objects" rather than "regular" objects).

By contrast, in the type erasure approach, there
are no inheritance relationships and value
semantics are employed. The fundamental idea is that
*any* class can "act as" an optimisation problem
as long as it conforms to a pre-determined interface.
Specifically, in the case of pagmo, any class that
implements a certain set of member functions can
be used to represent an optimisation problem.

In pagmo, type erasure is used pervasively not only
in the implementation of optimisation problems, but also
in the implementation of optimisation algorithms,
in the parallelisation strategies, in the migration framework,
etc. In the lingo of pagmo, we refer to classes that
conform to specific type-erased interfaces as *user-defined*
classes (e.g., user-defined problem, or UDP, user-defined
algorithm, or UDA, etc.).

.. [#coptlib] In C, where OOP is not natively supported,
   one can use function pointers to emulate some aspects of OOP.

Compiling the tutorials
-----------------------

The source code of the tutorials
can be found in the ``tutorials/`` directory in the pagmo
source tree. The tutorials can be compiled by enabling
the ``PAGMO_BUILD_TUTORIALS`` CMake option (see the
:ref:`installation instructions <install>`).
