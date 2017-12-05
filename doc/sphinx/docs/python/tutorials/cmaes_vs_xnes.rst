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
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from random import shuffle
    >>> 
    >>> trials = 100
    >>> bootstrap = 100
    >>> target = 1e-6
    >>> n_bins = 100
    >>> 
    >>> 
    >>> udp = pg.rosenbrock
    >>> dim = 20
    >>> prob = pg.problem(udp(dim))
    >>> experiments_list = [[pg.algorithm(pg.xnes(gen=4000)), 40],
    >>>                     [pg.algorithm(pg.cmaes(gen=4000)), 40],
    >>>                     [pg.algorithm(pg.xnes(gen=4000)), 60],
    >>>                     [pg.algorithm(pg.cmaes(gen=4000)), 60],
    >>>                     [pg.algorithm(pg.xnes(gen=4000)), 100],
    >>>                     [pg.algorithm(pg.cmaes(gen=4000)), 100],
    >>>                     ]
    >>> 
    >>> for algo, popsize in experiments_list:
    ...     runs = []
    ...     for i in range(trials):
    ...         pop = pg.population(prob, popsize)
    ...         pop = algo.evolve(pop)
    ...         runs.append([pop.problem.get_fevals(), pop.champion_f[0]])
    >>> 
    >>>     target_reached_at = []
    >>>     for i in range(bootstrap):
    ...         shuffle(runs)
    ...         tmp = [r[1] for r in runs]
    ...         t1 = np.array([min(tmp[:(i + 1)]) for i in range(len(tmp))])
    ...         t2 = np.cumsum([r[0] for r in runs])
    ...         idx = np.where(t1 < target)
    ...         target_reached_at.append(t2[idx][0])
    >>>     target_reached_at = np.array(target_reached_at)
    >>> 
    >>>     fevallim = 2 * max(target_reached_at)
    >>>     bins = np.linspace(0, fevallim, n_bins)
    >>>     ecdf = []
    >>>     for b in bins:
    ...         s = sum((target_reached_at) < b) / len(target_reached_at)
    ...         ecdf.append(s)
    >>>     plt.plot(bins, ecdf, label=algo.get_name().split(
    >>>         ":")[0] + " - " + str(popsize))
    >>> 
    >>> plt.legend()
    >>> ax = plt.gca()
    >>> ax.set_xscale('log')
    >>> plt.title(prob.get_name() + " - dimension " + str(dim))
    >>> plt.xlabel("N. fevals")



.. image:: ../../images/cmaes_vs_xnes1.png
   :scale: 40 %

.. image:: ../../images/cmaes_vs_xnes2.png
   :scale: 40 %

.. image:: ../../images/cmaes_vs_xnes3.png
   :scale: 40 %

.. image:: ../../images/cmaes_vs_xnes4.png
   :scale: 40 %