.. _py_tutorial_cec2013_copm:

Participating to the CEC2013 Competition
===============================================

In this tutorial we will show how to use pygmo to run algorithms on the test problem suite used in the
Special Session & Competition on Real-Parameter Single Objective Optimization at CEC-2013, Cancun, Mexico 21-23 June 2013

All of the cec2013 problems are box-bounded, continuous, single objective problems and are provided as UDP (user-defined
problems) by pygmo in the class :class:`~pygmo.core.cec2013`. Instantiating one of these problem is easy:

.. doctest::

    >>> import pygmo as pg
    >>> # The user-defined problem
    >>> udp = pg.cec2013(prob_id = 24, dim = 10)
    >>> # The pygmo problem
    >>> prob = pg.problem(udp)

as usual, we can quickly inspect the :class:`~pygmo.core.problem` printing it to screen:

.. doctest::

    >>> print(prob) #doctest: +NORMALIZE_WHITESPACE
    Problem name: CEC2013 - f24(cf04)
    	Global dimension:			10
    	Fitness dimension:			1
    	Number of objectives:			1
    	Equality constraints dimension:		0
    	Inequality constraints dimension:	0
    	Lower bounds: [-100, -100, -100, -100, -100, ... ]
    	Upper bounds: [100, 100, 100, 100, 100, ... ]
    <BLANKLINE>
    	Has gradient: false
    	User implemented gradient sparsity: false
    	Has hessians: false
    	User implemented hessians sparsity: false
    <BLANKLINE>
    	Function evaluations: 0
    <BLANKLINE>

Let us assume we want to assess the performance of (say) the optimization algorithm :class:`~pygmo.core.cmaes` (which
imlements as user-defined algorithm the Covariance Matrix Adaptation Evolutionary Strategy) on the whole
:class:`~pygmo.core.cec2013` problem suite at dimension D=2. Since the competition rules allowed D * 10000
function evaluation, we choose a population of 50 and 400 generations:

.. doctest::

    >>> # The cmaes pygmo algorithm
    >>> algo = pg.algorithm(pg.cmaes(gen=400, ftol=1e-9, xtol=1e-9))
    >>> # Defining all 28 problems dimension
    >>> D = 2
    >>> # Running the algo on them multiple times
    >>> error = []
    >>> trials = 25
    >>> for j in range(trials): # doctest: +SKIP
    ... 	for i in range(28):
    ... 		prob = pg.problem(pg.cec2013(prob_id = i+1, dim = D))
    ... 		pop = pg.population(prob,50)
    ... 		pop = algo.evolve(pop)
    ... 		error.append(pop.get_f()[pop.best_idx()] + 1400 - 100*i - 100*(i>13))

At the end of the script above, a matplotlib boxplot can be easily produced reporting the results:

.. doctest::

    >>> import matplotlib.pyplot as plt # doctest: +SKIP
    >>> res = plt.boxplot([error[s::28] for s in range(28)]) # doctest: +SKIP
    >>> plt.text(1, 105, algo.__repr__(), fontsize=8) # doctest: +SKIP
    >>> plt.ylim([-1,350]) # doctest: +SKIP
    >>> plt.title("CEC2013: dimension = 2") # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

.. image:: ../../images/cec2013_2_cmaes.png
    :scale: 100 %
    :alt: CEC2013-CMAES-2D
    :align: center

From the image above we see immediately that the problems from 1 to 20 are significantly easier to solve for
CMA-ES than the last eight. Note that the number of function evaluation used will, in this easy case, be
smaller than the allowed one as the algorithm may converge before the maximum number of generation allowed.

We may now try to do the same with a larger number of dimensions, say D=10. As we can now take advantage
of an increased number of function evaluations (100000), we set up CMA-ES differently (i.e. we use
a larger population and more maximum iterations):
