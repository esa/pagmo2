.. _py_tutorial_moo_moead:

================================================================
Using pygmo's :class:`~pygmo.core.moead` 
================================================================

In this tutorial we will see how to use the user-defined algorithm :class:`~pygmo.core.moead` 
provided by pygmo. In particular we will take as test cases, problems from the DTLZ suite implemented
in pygmo as the user-defined problem :class:`~pygmo.core.dtlz`.

The first, quick idea would be to instantiate the problem (say DTLZ1) and run :class:`~pygmo.core.moead`
with the default settings. We can monitor the convergence of the whole population to the Pareto front
using the :func:`~pygmo.core.dtlz.p_distance` metric.


.. image:: ../../images/mo_dtlz_moead_grid_tch.png
   :scale: 60 %
   :alt: dtlz1 moead grid tchebycheff
   :align: right

.. doctest::
   
    >>> from pygmo import *
    >>> udp = dtlz(id = 1)
    >>> pop = population(prob = udp, size = 105)
    >>> algo = algorithm(moead(gen = 100))
    >>> for i in range(10):
    ...     pop = algo.evolve(pop)
    >>> print(udp.p_distance(pop)) # doctest: +SKIP
    0.0012264939631066003

Since the :func:`~pygmo.core.dtlz.p_distance` does not capture the information on the spread of the solutions we
also compute the hypervolume indicator using the pygmo class :class:`~pygmo.core.hypervolume`:

.. doctest::

    >>> hv = hypervolume(pop)
    >>> hv.compute(ref_point = [1.,1.,1.]) # doctest: +ELLIPSIS
    0.9...

In this case, the reference point can be set manually as the dtlz1 problem is well known. We can also visualize the 
whole population as the user-defined problem :class:`~pygmo.core.dtlz` implements a plot functionality:

.. doctest::
   
    >>> from matplotlib import pyplot as plt # doctest: +SKIP
    >>> udp.plot(pop) # doctest: +SKIP
    >>> plt.title("DTLZ1 - MOEAD - GRID - TCHEBYCHEFF") # doctest: +SKIP

We have used the default parameters of :class:`~pygmo.core.moead` in obtaining the results above. In 
particular the **weight_generation** kwarg was set to **grid** and the **decomposition** kwarg was set to
**tchebycheff**, as can be easily seen inspecting as follows:

.. doctest::

    >>> print(algo) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Algorithm name: MOEA/D - DE [stochastic]
    Extra info:
        Generations: 100
        Weight generation: grid
        Decomposition method: tchebycheff
        Neighbourhood size: 20
        Parameter CR: 0.5
        Parameter F: 0.5
        Distribution index: 20
        Chance for diversity preservation: 0.9
        Seed: ...
        Verbosity: 0

The **weight_generation** method **grid** offers a uniform distribution of the decomposed weights, but is 
limiting the population size as it only allows for certaing sizes accoridng to the number of objectives. This 
can reveal to be limiting when using :class:`~pygmo.core.moead` in comparisons or in other advanced setups. In these 
cases pygmo provides alternative methods for weight generation. In particular, the original **low discrepancy** method
makes sure that any number of weights can be generated while ensuring a low discrepancy spread over the objective space.

The **decomposition** method **tchebycheff** can also be changed as pygmo implements the boundary intersection method too
which, when applicable, results in a better spread of the final population over the Pareto front. Repeating the optimization above
with different instances of :class:`~pygmo.core.moead` results in the plots below.

.. image:: ../../images/mo_dtlz_moead_array.png
   :align: center