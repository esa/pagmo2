.. _py_tutorial_cmaes_vs_xnes:

A comparison between CMA-ES and xNES
===============================================

In this tutorial we will show how to use pygmo to comapre the performances of the two UDAs: :class:`~pygmo.cmaes` and :class:`~pygmo.xnes` 

Both algorithms are based on the idea of updating a gaussian distribution that regulates the generation of new samples, but they
differ from the update rules, the ones of xNES being arguably more elegant and surely more compact.

Let us start setting up the algorithms. Both :class:`~pygmo.cmaes` and :class:`~pygmo.xnes` are set up to terminate
with the same convergence requirements and starting distribution (i.e. equal ``sigma0``). The initial population size is
set to be :math:`n = 4 + log(3 d)` where `d` is the problem dimension.

.. doctest::

    >>> import pygmo as pg
    >>> from math import floor
    >>> from matplotlib import pyplot as plt # doctest: +SKIP
    >>> import matplotlib.lines as mlines # doctest: +SKIP
    >>> # We instantiate the algorithms
    >>> xnes = pg.algorithm(pg.xnes(gen = 40000, ftol=1e-8, xtol=1e-14, sigma0 = 0.25))
    >>> cmaes = pg.algorithm(pg.cmaes(gen = 40000, ftol=1e-8, xtol=1e-14, sigma0 = 0.25))
    >>> # We define the udp
    >>> udp = pg.rosenbrock
    >>> # We will test on the following problem dimensions
    >>> dims = [2, 4, 8, 10, 20, 30]
    >>> # And using the following population sizes
    >>> sizes = [4 + floor(3 * d) for d in dims]

We then define an auxiliary function that will run the experiments recording the number of function
evaluations upon convergence and the reached objective function value:

    >>> def run_experiments(a, trials):
    >>>     fevals = []
    >>>     values = []
    >>>     for dim in dims:
    >>>         tmp = []
    >>>         tmp_val = []
    >>>         prob = pg.problem(udp(dim = dim))
    >>>         for i in range(trials):
    >>>             pop = pg.population(prob,10)
    >>>             pop = a.evolve(pop)
    >>>             tmp.append(pop.problem.get_fevals())
    >>>             tmp_val.append(pop.champion_f[0])
    >>>         fevals.append(tmp)
    >>>         values.append(tmp_val)
    >>>     return fevals, values

Clearly we could use a :class:`~pygmo.archipelago` to parallelize these experiments, but for the purpose of this tutorial
we will leave things as they are to not 

    >>> colors = ["red","blue"]
    >>> handles = []
    >>> trials = 100
    >>> 
    >>> for a, color in zip([xnes, cmaes], colors): # doctest: +SKIP
    >>>     res, values = run_experiments(a) 
    >>>     print(a.get_name())
    >>>     for v in values:
    >>>         print("Failed: ", sum(array(v)>1e-7))
    >>>     bp = plt.boxplot(res)
    >>>     for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    >>>             plt.setp(bp[element], color=color)
    >>>     handles.append(mlines.Line2D([], [], color=color, label=a.get_name().split(":")[0]))
    >>> plt.legend(handles=handles, loc=2) # doctest: +SKIP
    >>> ax  =plt.gca() # doctest: +SKIP
    >>> ax.set_yscale('log') # doctest: +SKIP
    >>> ax.set_xticklabels(dims) # doctest: +SKIP
    >>> plt.xlabel("Problem dimension") # doctest: +SKIP
    >>> plt.ylabel("Function evaluations") # doctest: +SKIP
    >>> plt.title(pg.problem(udp()).get_name()) # doctest: +SKIP
    >>> plt.ion() # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

.. image:: ../../images/cmaes_vs_xnes1.png
   :scale: 60 %

.. image:: ../../images/cmaes_vs_xnes2.png
   :scale: 60 %

.. image:: ../../images/cmaes_vs_xnes3.png
   :scale: 60 %

.. image:: ../../images/cmaes_vs_xnes4.png
   :scale: 60 %