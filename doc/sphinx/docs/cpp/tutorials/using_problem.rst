.. _cpp_tutorial_using_problem:

Use of the class :class:`~pagmo::problem`
=============================================

.. image:: ../../images/prob_no_text.png

The :class:`~pagmo::problem` class represents a generic optimization
problem. The user codes the details of such a problem in a separate class (the
user-defined problem, or UDP) which is then passed to :class:`~pagmo::problem`
that provides a single unified interface.

.. note:: The User Defined Problems (UDPs) are optimization problems (coded by the user) that can
          be used to build a pagmo object of type :class:`~pagmo::problem`

Some UDPs (optimization problems) are already provided with pagmo and we refer to them as pagmo UDPs.

For the purpose of this tutorial we are going to use a pagmo UDP called :class:`~pagmo::rosenbrock`
to show the basic construction of a :class:`~pagmo::problem`, but the same logic would also
apply to a custom UDPs, that is a UDP that is actually coded by the user.

The following code snippets can either be used in a compiled file or executed in a Jupyter notebook.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/esa/pagmo2-binder/master?filepath=Basic%20Problem.ipynb


Let us start:

.. code-block:: c++

    #include <iostream>
    #include <pagmo/pagmo.hpp>
    namespace pg = pagmo;
    auto prob = pg::problem(pg::rosenbrock(5));
    std::cout << prob << std::endl;

This will result in an output close to the following:

.. code-block:: none

    Problem name: Multidimensional Rosenbrock Function
        Global dimension:           5
        Integer dimension:          0
        Fitness dimension:          1
        Number of objectives:           1
        Equality constraints dimension:     0
        Inequality constraints dimension:   0
        Lower bounds: [-5, -5, -5, -5, -5]
        Upper bounds: [10, 10, 10, 10, 10]
        Has batch fitness evaluation: false
    
        Has gradient: true
        User implemented gradient sparsity: false
        Expected gradients: 5
        Has hessians: false
        User implemented hessians sparsity: false
    
        Fitness evaluations: 0
        Gradient evaluations: 0
    
        Thread safety: constant


In the code above, after the trivial import of the pagmo package, we define a variable prob
by constructing a :class:`~pagmo::problem` from :class:`~pagmo::rosenbrock`, a multidimensional problem
constructed from its global dimensions. In the following line we print the :class:`~pagmo::problem`
We can see, at a glance, that the UDP :class:`~pagmo::rosenbrock` is a five dimensional, single objective, box constrained
problem which has a gradient but for which neither hessians nor sparsity information is provided in the UDP.

We also see that its fitness function has never been called hence the counter for fitness evaluations is
zero.

All of the information contained in the :class:`~pagmo::problem` print out can be retrieved using
the appropriate methods, for example:

.. code-block:: c++

    prob.get_fevals()

Output:

.. code-block:: none
    
    0

Lets check how a fitness computation increases the counter:

.. code-block:: c++

    prob.fitness([1,2,3,4,5])

Output:

.. code-block:: none
    
    array([14814.])

The number of evaluations has now increased:

.. code-block:: c++

    prob.get_fevals()

Output:

.. code-block:: none
    
    1

We may also get back a const pointer to the UDP, and thus access all the methods not exposed in the
:class:`~pagmo::problem` interface at any time via the templated :func:`~pagmo::problem.extract()` method:

.. code-block:: c++

    auto udp = prob.extract<pg::rosenbrock>()

Such an *extraction* will only work if the correct UDP type is passed as template parameter.
