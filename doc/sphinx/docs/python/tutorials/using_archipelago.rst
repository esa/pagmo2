.. _py_tutorial_using_archipelago:

Use of the class :class:`~pygmo.archipelago`
===============================================

The :class:`~pygmo.archipelago` class is the main parallelization engine of pagmo. It essentially is
a container of :class:`~pygmo.island` able to initiate evolution (optimization tasks) in each :class:`~pygmo.island`
asynchronously while keeping track of the results and of the information exchange (migration) between the tasks (via the
generalized island model). The various :class:`~pygmo.island` in an :class:`~pygmo.archipelago` can be heterogeneous
and refer to different UDAs, UDPs and UDIs.

In this tutorial we will show how to use an :class:`~pygmo.archipelago` to run in parallel several
optimization tasks defined in a UDP. So let us start defining our UDP. We will here define a single objective
problem with one equality and one inequality constraint. The problem in itself is not important for the
purpose of this tutorial. Its mathematical definition is:

.. math::
   \begin{array}{ll}
     \mbox{minimize: } & \sum_i x_i \\
     \mbox{subject to:} & -1 \le x_i \le 1, i = 1..n \\
                        & \sum_i x_i^2 = 1 \\
                        & \sum_i x_i \ge 0 \\
   \end{array}

and the optimal value for the objectove function is 0. We can write the above problem as a pagmo UDP (see:
:ref:`py_tutorial_coding_udp_simple` and :ref:`py_tutorial_coding_udp_constrained` to learn how UDPs are
defined from Python)

.. doctest::

    >>> class toy_problem:
    ...     def __init__(self, dim):
    ...         self.dim = dim
    ...
    ...     def fitness(self, x):
    ...         return [sum(x), 1 - sum(x*x), - sum(x)]
    ...
    ...     def gradient(self, x):
    ...         return pg.estimate_gradient(lambda x: self.fitness(x), x) # numerical gradient
    ...
    ...     def get_nec(self):
    ...         return 1
    ...
    ...     def get_nic(self):
    ...         return 1
    ...
    ...     def get_bounds(self):
    ...         return ([-1] * self.dim, [1] * self.dim)
    ...
    ...     def get_name(self):
    ...         return "A toy problem"
    ...
    ...     def get_extra_info(self):
    ...         return "\tDimensions: " + str(self.dim)

Now, without further ado, lets use the full power of pygmo and prepare to be shocked:

.. doctest::

    >>> import pygmo as pg
    >>> a_cstrs_sa = pg.algorithm(pg.cstrs_self_adaptive(iters=2000))
    >>> p_toy = pg.problem(toy_problem(100))
    >>> p_toy.c_tol = [1e-4, 1e-4]
    >>> archi = pg.archipelago(n=8,algo=a_cstrs_sa, prob=p_toy, pop_size=100)
    >>> print(archi) #doctest: +SKIP
    Number of islands: 8
    Status: idle
    <BLANKLINE>
    Islands summaries:
    <BLANKLINE>
        #  Type                    Algo                                Prob           Size  Status  
        --------------------------------------------------------------------------------------------
        0  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   idle    
        1  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   idle    
        2  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   idle    
        3  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   idle    
        4  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   idle    
        5  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   idle    
        6  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   idle    
        7  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   idle    

To instantiate the :class:`~pygmo.archipelago` we have used the constructor from a ``problem`` and ``algorithm``.
This leaves to the ``archi`` object the task of building the various populations and islands. To do so, several
choices are made for the user, starting with the island type. In this case, as seen from the screen
printout of the ``archi`` object, 8 Multiprocessing islands are being used. To control what islands will
assemble an archipelago, the user can instantiate an empty archipelago and use the
:func:`pygmo.archipelago.push_back()` method to insert one by one all the islands. 

.. note::
   The island type selected by the :class:`~pygmo.archipelago` constructor used is the ``Multiprocessing island``, 
   (:class:`pygmo.py_islands.mp_island`) as we run this example on py36 and a linux machine. In general, the exact island used is platform (a dn pop and algo)
   dependent and is described in the docs of the :class:`~pygmo.island` class constructor.

After inspection, let us now run the evolution.

 .. doctest::

    >>> archi.get_champions_f() #doctest: +SKIP
    [array([  2.18826798, -26.60899368,  -2.18826798]),
    array([  3.39497588, -24.48739141,  -3.39497588]),
    array([  2.3240917 , -26.88225527,  -2.3240917 ]),
    array([  0.13134093, -28.47299705,  -0.13134093]),
    array([  6.53062434, -24.98724057,  -6.53062434]),
    array([  1.02894159, -25.69765425,  -1.02894159]),
    array([  4.07802374, -23.82020921,  -4.07802374]),
    array([  1.71396489, -25.90794514,  -1.71396489])]
    >>> archi.evolve()
    >>> print(archi) #doctest: +SKIP
    Number of islands: 8
    Status: busy
    <BLANKLINE>
    Islands summaries:
    <BLANKLINE>
        #  Type                    Algo                                Prob           Size  Status  
        --------------------------------------------------------------------------------------------
        0  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   busy    
        1  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   busy    
        2  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   busy    
        3  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   busy    
        4  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   busy    
        5  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   busy    
        6  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   busy    
        7  Multiprocessing island  Self-adaptive constraints handling  A toy problem  100   busy   

Note how the evolution happening on the various islands does not interfere with our main process 
as as it happens asynchronously on separate threads. We then have to call the :func:`pygmo.archipelago.wait()` 
method to have the main process explicitly wait for all islands to be finished.

 .. doctest::

    >>> archi.wait()
    >>> archi.get_champions_f() #doctest: +NORMALIZE_WHITESPACE
    [array([ 0.01219149,  0.00462767, -0.01219149]),
    array([ 0.00012038,  0.00107938, -0.00012038]),
    array([  2.47246441e-02,   6.77824904e-05,  -2.47246441e-02]),
    array([ 0.00791089,  0.00311376, -0.00791089]),
    array([  4.85915841e-03,  -3.74569709e-10,  -4.85915841e-03]),
    array([ 0.00752764, -1.1718177 , -0.00752764]),
    array([  5.29472090e-02,  -3.39153972e-09,  -5.29472090e-02]),
    array([ 0.07301779, -0.56072333, -0.07301779])]

Different islands produce different results, in this case, as the various populations and algorithms where
constructed using random seeds.