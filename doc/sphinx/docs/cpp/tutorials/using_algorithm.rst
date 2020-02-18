.. _py_tutorial_using_algorithm:

Use of the class :class:`~pagmo::algorithm`
===============================================

.. image:: ../../images/algo_no_text.png

The :class:`~pagmo::algorithm` class represents a generic optimization
algorithm. The user codes the details of such an algorithm in a separate class (the
user-defined algorithm, or UDA) which is then passed to :class:`~pagmo::algorithm`
that provides a single unified interface.

.. note::  The User Defined Algorithms (UDAs) are optimization algorithms (coded by the user) that can
           be used to build a pagmo object of type :class:`~pagmo::algorithm`

Some UDAs (optimization algorithms) are already provided with pagmo and we refer to them as pagmo UDAs.

For the purpose of this tutorial we are going to use a pagmo UDA called :class:`~pagmo::cmaes`
to show the basic construction of an :class:`~pagmo::algorithm`, but the same logic would also
apply to a custom UDAs, that is a UDA that is actually coded by the user.

The following code snippets can either be used in a compiled file or executed in a Jupyter notebook.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/esa/pagmo2-binder/master?filepath=Basic%20Problem.ipynb

Let us start:

.. code-block:: c++

    #include <pagmo/pagmo.hpp>
    #include <iostream>
    namespace pg = pagmo;

    unsigned gen = 100.0;
    double sigma0 = 0.3;
    double cc, cs, c1, cmu, ftol, xtol;
    std::tie(cc, cs, c1, cmu, ftol, xtol) = std::make_tuple(-1, -1, -1, -1, 1e-6, 1e-6);
    bool memory = false;
    bool force_bounds = false;

    auto algo = pg::algorithm(pg::cmaes(gen, cc, cs, c1, cmu, sigma0, ftol, xtol, memory, force_bounds));
    
    std::cout << algo;

This will result in an output close to the following:

.. code-block:: none

    Algorithm name: CMA-ES: Covariance Matrix Adaptation Evolutionary Strategy [stochastic]
    	Thread safety: basic
    
    Extra info:
    	Generations: 100
    	cc: auto
    	cs: auto
    	c1: auto
    	cmu: auto
    	sigma0: 0.3
    	Stopping xtol: 1e-06
    	Stopping ftol: 1e-06
    	Memory: false
    	Verbosity: 0
    	Force bounds: false
    	Seed: ...

In the code above, after the trivial import of the pagmo package, we define a variable algo
by constructing an :class:`~pagmo::algorithm` from :class:`~pagmo::cmaes`, our implementation
of the Covariance Matrix Adaptation Evolutionary Strategy. To construct the pagmo UDA we also pass
some parameters whose meaning is documented in :class:`~pagmo::cmaes`.
In the following line we inspect the :class:`~pagmo::algorithm`. We can see, at a glance, the
name of the :class:`~pagmo::algorithm` and some extra info that indicate the user (in this case us),
has implemented, in the UDA (in this case :class:`~pagmo::cmaes`), the optional method
``get_extra_info()`` that prints to screen some fundamental parameters defining the UDA.

We may also get access to the UDA, and thus to its methods not exposed in the
:class:`~pagmo::algorithm` interface, at any time via the :class:`~pagmo::algorithm.extract` method:

.. code-block:: c++

    auto uda = algo.extract<pg::cmaes>();

Such an extraction will only work if the correct UDA type is passed as template parameter.
