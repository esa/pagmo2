.. _py_tutorial_using_archipelago:

Use of the class :class:`~pygmo.archipelago`
===============================================

.. image:: ../../images/archi_no_text.png

The :class:`~pygmo.archipelago` class is the main parallelization engine of pygmo. It essentially is
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

and the optimal value for the objective function is 0. We can write the above problem as a pygmo UDP (see:
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
    >>> a_cstrs_sa = pg.algorithm(pg.cstrs_self_adaptive(iters=1000))
    >>> p_toy = pg.problem(toy_problem(50))
    >>> p_toy.c_tol = [1e-4, 1e-4]
    >>> archi = pg.archipelago(n=32,algo=a_cstrs_sa, prob=p_toy, pop_size=70)
    >>> print(archi) #doctest: +SKIP
    Number of islands: 32
    Status: idle
    <BLANKLINE>
    Islands summaries:
    <BLANKLINE>
        #  Type                    Algo                                Prob           Size  Status  
        --------------------------------------------------------------------------------------------
        0  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    idle    
        1  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    idle    
        2  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    idle    
        3  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    idle    
        4  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    idle    
        5  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    idle    
        6  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    idle    
        7  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    idle 
        ...  

To instantiate the :class:`~pygmo.archipelago` we have used the constructor from a ``problem`` and ``algorithm``.
This leaves to the ``archi`` object the task of building the various populations and islands. To do so, several
choices are made for the user, starting with the island type. In this case, as seen from the screen
printout of the ``archi`` object, Multiprocessing islands are being used. To control what islands will
assemble an archipelago, the user can instantiate an empty archipelago and use the
:func:`pygmo.archipelago.push_back()` method to insert one by one all the islands. 

.. note::
   The island type selected by the :class:`~pygmo.archipelago` constructor is, in this case, the
   ``multiprocessing island``, (:class:`pygmo.mp_island`) as we run this example on py36 and
   a linux machine. In general, the exact island chosen is platform, population and algorithm
   dependent and such choice is described in the docs of the :class:`~pygmo.island` class constructor.

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
    array([  1.71396489, -25.90794514,  -1.71396489]),
    ...]
    >>> archi.evolve() #doctest: +SKIP
    >>> print(archi) #doctest: +SKIP
    Number of islands: 32
    Status: busy
    <BLANKLINE>
    Islands summaries:
    <BLANKLINE>
        #  Type                    Algo                                Prob           Size  Status  
        --------------------------------------------------------------------------------------------
        0  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    busy    
        1  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    busy    
        2  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    busy    
        3  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    busy    
        4  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    busy    
        5  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    busy    
        6  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    busy    
        7  Multiprocessing island  Self-adaptive constraints handling  A toy problem  70    busy   
        ...

Note how the evolution is happening in parallel on 32 separate threads (each one spawning a process, in this case, as a multiprocessing island is used).
The evolution happens asynchronously and thus does not interfere directly with our main process. We then have to call the :func:`pygmo.archipelago.wait()` 
method to have the main process explicitly wait for all islands to be finished.

.. doctest::

    >>> archi.wait()
    >>> archi.get_champions_f() #doctest: +SKIP
    [array([  1.16514064e-02,   4.03450637e-05,  -1.16514064e-02]),
    array([ 0.02249111,  0.00392739, -0.02249111]),
    array([  6.09564060e-03,  -4.93961313e-05,  -6.09564060e-03]),
    array([ 0.01161224, -0.00815189, -0.01161224]),
    array([ -1.90431378e-05,   7.65501702e-05,   1.90431378e-05]),
    array([ -5.45044897e-05,   5.70199057e-05,   5.45044897e-05]),
    array([ 0.00541601, -0.08208163, -0.00541601]),
    array([ -6.95677113e-05,  -7.42268924e-05,   6.95677113e-05]),
    array([ 0.00335729,  0.00684969, -0.00335729]),
    ...


Different islands produce different results, in this case, as the various populations and algorithms where
constructed using random seeds.

.. note::
   The use of the UDA :class:`~pygmo.cstrs_self_adaptive` is here only chosen to keep the various threads occupied,
   most likely other local algorithm would, in this case, be a superior choice.

Managing exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
What happens if, during the optimization task sent to an :class:`~pygmo.island`, an exception happens? This question is already explored 
in the :ref:`island tutorial<py_tutorial_using_island>` and since an :class:`~pygmo.archipelago` is, basically, a container for multiple :class:`~pygmo.island`
here we will overlap with part of that tutorial, exploring exceptions thrown in the :class:`~pygmo.archipelago` context.

To show how pygmo handles these situations we use the fake problem below throwing as soon as 300 fitness evaluations are made.

    >>> class raise_exception:
    ...     def __init__(self):
    ...         self.counter=0;
    ...     def fitness(self,dv):
    ...         if self.counter == 300:
    ...             raise
    ...         self.counter += 1
    ...         return [0]
    ...     def get_bounds(self):
    ...         return ([0],[1])
    ...     def get_name(self):
    ...         return "A throwing UDP"

Let us now instantiate and run a :class:`~pygmo.archipelago`:

    >>> archi = pg.archipelago(n = 5, algo = pg.simulated_annealing(Ts = 10, Tf = 0.1, n_T_adj  = 40), prob = raise_exception(), pop_size = 20)
    >>> archi.evolve() #doctest: +SKIP
    >>> archi.wait()
    >>> print(archi) #doctest: +SKIP
    Number of islands: 5
    Status: idle - **error occurred**
    <BLANKLINE>
    Islands summaries:
    <BLANKLINE>
        #  Type                    Algo                            Prob            Size  Status                     
        ------------------------------------------------------------------------------------------------------------
        0  Multiprocessing island  Simulated Annealing (Corana's)  A throwing UDP  20    idle - **error occurred**  
        1  Multiprocessing island  Simulated Annealing (Corana's)  A throwing UDP  20    idle - **error occurred**  
        2  Multiprocessing island  Simulated Annealing (Corana's)  A throwing UDP  20    idle - **error occurred**  
        3  Multiprocessing island  Simulated Annealing (Corana's)  A throwing UDP  20    idle - **error occurred**  
        4  Multiprocessing island  Simulated Annealing (Corana's)  A throwing UDP  20    idle - **error occurred**  

The print statment allows us to visually see that errors occured in all islands (each one is containing a copy of the problem and thus 
starts at 0 fitness evaluations to then hit the 300 fevals triggering the ``raise`` statement).
We do not know at this stage what happened exactly as the exception is thrown on a separate thread/process and thus its hidden from our main
scope. To inspect what happened we can write:


    >>> archi.wait_check() #doctest: +SKIP
    RuntimeError                              Traceback (most recent call last)
    <ipython-input-22-d8562cc51573> in <module>()
    ----> 1 archi.wait_check()
    <BLANKLINE>
    RuntimeError: The asynchronous evolution of a pythonic island of type 'Multiprocessing island' raised an error:
    multiprocessing.pool.RemoteTraceback: 
    """
    Traceback (most recent call last):
    File "/home/dario/miniconda3/envs/pagmo/lib/python3.6/multiprocessing/pool.py", line 119, in worker
        result = (True, func(*args, **kwds))
    File "/home/dario/miniconda3/envs/pagmo/lib/python3.6/site-packages/pygmo/_py_islands.py", line 40, in _evolve_func
        return algo.evolve(pop)
    File "<ipython-input-19-f5ede4c63180>", line 6, in fitness
    RuntimeError: No active exception to reraise
    """
    <BLANKLINE>
    The above exception was the direct cause of the following exception:
    <BLANKLINE>
    Traceback (most recent call last):
    File "/home/dario/miniconda3/envs/pagmo/lib/python3.6/site-packages/pygmo/_py_islands.py", line 128, in run_evolve
        return res.get()
    File "/home/dario/miniconda3/envs/pagmo/lib/python3.6/multiprocessing/pool.py", line 608, in get
        raise self._value
    RuntimeError: No active exception to reraise

which will rethrow the first encountered exception and reset all the island states to ``idle``.