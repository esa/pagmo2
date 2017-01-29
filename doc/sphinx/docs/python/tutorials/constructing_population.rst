.. py_tutorial_using_pygmo_UDAs

Constructing a population
==============================

In pygmo, a :class:`~pygmo.core.population` is a storage for candidate solutions
to some :class:`~pygmo.core.problem` (sometimes called individuals).
It thus contains a :class:`~pygmo.core.problem` and a number of decision
vectors (chromosomes) and fitness vectors, each associated to a unique ID as
to allow some degree of tracking.

.. doctest::

    >>> import pygmo as pg
    >>> prob = pg.problem(pg.rosenbrock(dim = 4))
    >>> pop1 = pg.population(prob)
    >>> pop2 = pg.population(prob, size = 5, seed= 723782378)

In the code above, after the trivial import of the pygmo package, we first define a :class:`~pygmo.core.problem`
from :class:`~pygmo.core.rosenbrock`, and then we construct two populations (i.e. two sets of candidate solutions).
The first population, ``pop1``, is empty, while the second population is initialized with
five candidate solutions, hence also causing five fitness evaluations. The candidate solutions in
``pop2`` are randomly created within the box-bounds of the :class:`~pygmo.core.problem` using
the random seed passed as kwarg.

.. doctest::

    >>> print(pop1.size())
    0
    >>> print(pop1.get_problem().get_fevals())
    0
    >>> print(pop2.size())
    5
    >>> print(pop2.get_problem().get_fevals())
    5

The full inspection of a :class:`~pygmo.core.population` is possible, as usual,
via its methods or glancing to the screen print of the entire class:

.. doctest::

    >>> print(pop2) #doctest: +SKIP
    ...
    <BLANKLINE>
    List of individuals:
    #0:
    	ID:			10207561574636104601
    	Decision vector:	[-0.777137, 7.91467, -4.31933, 5.92765]
    	Fitness vector:		[470010]
    #1:
    	ID:			15637512834538262525
    	Decision vector:	[8.94985, 0.924838, 4.39905, -1.17683]
    	Fitness vector:		[670339]
    #2:
    	ID:			15983142956381075935
    	Decision vector:	[-0.291054, 4.99031, 1.34008, -0.00609471]
    	Fitness vector:		[58271.1]
    #3:
    	ID:			9198455452901607935
    	Decision vector:	[2.18419, -1.04727, 6.35101, 6.39632]
    	Fitness vector:		[121365]
    #4:
    	ID:			4553447107323210017
    	Decision vector:	[7.50729, -1.14561, 5.98053, -3.48833]
    	Fitness vector:		[487030]

Individuals, i.e. new candidate solutions can be put into a population calling
its ``push_pack`` method:

.. doctest::

    >>> pop1.push_back([0.1,0.2,0.3,0.4]) # correct size
    >>> pop1.size()
    1
    >>> pop1.get_problem().get_fevals()
    1
    >>> pop1.push_back([0.1,0.2,0.3]) # wrong size
    Traceback (most recent call last):
      File ".../lib/python3.6/doctest.py", line 1330, in __run
        compileflags, 1), test.globs)
      File "<doctest default[3]>", line 1, in <module>
        pop1.push_back([0.1,0.2,0.3])
    ValueError:
    function: check_decision_vector
    where: /Users/darioizzo/Documents/pagmo2/include/pagmo/problem.hpp, 1835
    what: Length of decision vector is 3, should be 4

Consistency checks are done by ``push_pack``, e.g. on the decision vector
length.

.. note:: Decision vectors that are outside of the box bounds are allowed to be
          pushed back into a population

When designing user-defined algorithms (UDAs) it is often important to be able to change
some individual decision vector:

.. doctest::

    >>> pop1.get_problem().get_fevals()
    1
    >>> print(pop1.get_x()[0])
    [ 0.1  0.2  0.3  0.4]
    >>> pop1.set_x(0, [1.,2.,3.,4.])
    >>> pop1.get_problem().get_fevals()
    2
    >>> print(pop1.get_f()[0])
    [ 2705.]
    >>> pop1.set_xf(0, [1.,2.,3.,4.], [8.43469444])
    >>> pop1.get_problem().get_fevals()
    2
    >>> print(pop1.get_f()[0])
    [ 8.43469444]

.. note:: Using the method ``set_xf`` it is possible to avoid triggering the fitness
          function evaluation, but it is also possible to inject spurious information
          into the population (i.e. breaking the relation between decision vectors
          and fitness vectors imposed by the problem)
