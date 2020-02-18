.. _py_tutorial_using_population:

Use of the class :class:`~pagmo::population`
================================================

.. image:: ../../images/pop_no_text.png

In pagmo, a :class:`~pagmo::population` is a storage for candidate solutions
to some :class:`~pagmo::problem` (sometimes called individuals).
It thus contains a :class:`~pagmo::problem` and a number of decision
vectors (chromosomes) and fitness vectors, each associated to a unique ID as
to allow some degree of tracking.

.. code-block:: c++

    #include <pagmo/pagmo.hpp>
    #include <iostream>

    namespace pg = pagmo;
    auto prob = pg::problem(pg::rosenbrock(4));
    auto pop1 = pg::population(prob)
    auto pop2 = pg::population(prob, 5, 723782378)

In the code above, after the trivial import of the pagmo package, we first define a :class:`~pagmo::problem`
from :class:`~pagmo::rosenbrock`, and then we construct two populations (i.e. two sets of candidate solutions).
The first population, ``pop1``, is empty, while the second population is initialized with
five candidate solutions, hence also causing five fitness evaluations. The candidate solutions in
``pop2`` are randomly created within the box-bounds of the :class:`~pagmo::problem` using
the random seed passed as third argument.

.. code-block:: c++

    std::cout << pop1.size() << std::endl;
    std::cout << pop1.get_problem().get_fevals() << std::endl;
    std::cout << pop2.size() << std::endl;
    std::cout << pop2.get_problem().get_fevals() << std::endl;

Output:

.. code-block:: none

    0
    0
    5
    5

The full inspection of a :class:`~pagmo::population` is possible, as usual,
via its methods or glancing to the screen print of the entire class:

.. code-block:: c++

    std::cout << pop2;

Output:

.. code-block:: none

    Problem name: Multidimensional Rosenbrock Function
        Global dimension:           4
        Integer dimension:          0
        Fitness dimension:          1
        Number of objectives:           1
        Equality constraints dimension:     0
        Inequality constraints dimension:   0
        Lower bounds: [-5, -5, -5, -5]
        Upper bounds: [10, 10, 10, 10]
        Has batch fitness evaluation: false

        Has gradient: true
        User implemented gradient sparsity: false
        Expected gradients: 4
        Has hessians: false
        User implemented hessians sparsity: false

        Fitness evaluations: 5
        Gradient evaluations: 0

        Thread safety: constant

    Population size: 5

    List of individuals: 
    #0:
        ID:         15730941710914891558
        Decision vector:    [-0.777137, 7.91467, -4.31933, 5.92765]
        Fitness vector:     [470010]
    #1:
        ID:         4004245315934230679
        Decision vector:    [3.38547, 8.94985, 0.924838, 4.39905]
        Fitness vector:     [628823]
    #2:
        ID:         12072501637330415325
        Decision vector:    [-1.17683, 1.16786, -0.291054, 4.99031]
        Fitness vector:     [2691.53]
    #3:
        ID:         15298104717675893584
        Decision vector:    [1.34008, -0.00609471, -2.80972, 2.18419]
        Fitness vector:     [4390.61]
    #4:
        ID:         4553447107323210017
        Decision vector:    [-1.04727, 6.35101, 6.39632, 5.80792]
        Fitness vector:     [241244]

    Champion decision vector: [-1.17683, 1.16786, -0.291054, 4.99031]
    Champion fitness: [2691.53]


Individuals, i.e. new candidate solutions can be put into a population calling
its :func:`~pagmo::population.push_back()` method:

.. code-block:: c++

    pop1.push_back({0.1,0.2,0.3,0.4}); // correct size
    pop1.size();
    pop1.get_problem().get_fevals()

Output:

.. code-block:: none

    1
    1

Some consistency checks are done by :func:`~pagmo::population.push_back()`, e.g. on the decision vector
length.

.. code-block:: c++

    pop1.push_back({0.1,0.2,0.3}) // wrong size

Output:

.. code-block:: none

    Standard Exception: 
    function: prob_check_dv
    where: /home/conda/feedstock_root/build_artifacts/pagmo_1579180152081/work/src/problem.cpp, 916
    what: A decision vector is incompatible with a problem of type 'Multidimensional Rosenbrock Function': the number of dimensions of the problem is 4, while the decision vector has a size of 3 (the two values should be equal)
    

.. note:: Decision vectors that are outside of the box bounds are allowed to be
          pushed back into a population

The snippet above will trigger fitness function evaluations as the decision vector is always associated to a
fitness vector in a :class:`~pagmo::population`. If the fitness vector associated to a chromosome is known,
you may also push it back in a population and avoid triggering a fitness re-evaluation by typing:

.. code-block:: c++

    pop1.push_back({0.2,0.3,1.3,0.2}, {11.2})
    pop1.get_problem().get_fevals()

Output:

.. code-block:: none

    1


When designing user-defined algorithms (UDAs) it is often important to be able to change
some individual decision vector:

.. code-block:: c++

    pop1.get_x()[0];
    
    pop1.set_x(0,{1.,2.,3.,4.})
    pop1.problem.get_fevals()

Output:

.. code-block:: none

    [0.1  0.2  0.3  0.4]
    2

Again, the fitness evaluation can be avoided if the fitness is known:

.. code-block:: c++

    pop1.get_f()[0]
    pop1.set_xf(0, {1.,2.,3.,4.}, {8.43469444})
    pop1.get_problem().get_fevals()
    pop1.get_f()[0]

Output:

.. code-block:: none
    
    { 2705.0000 }
    2
    { 8.4346944 }

.. note:: Using the method :func:`~pagmo::population.set_xf()` or:func:`~pagmo::population.push_back()` it is possible to avoid
          triggering fitness function evaluations, but it is also possible to inject
          spurious information into the population (i.e. breaking the relation between
          decision vectors and fitness vectors imposed by the problem)

The best individual in a population can be extracted as:

.. code-block:: c++

    // The decision vector
    pop1.get_x()[pop1.best_idx()]
    
    // The fitness vector
    pop1.get_f()[pop1.best_idx()]

Output:

.. code-block:: none

    { 1.0000000, 2.0000000, 3.0000000, 4.0000000 }
    { 8.4346944 }
    

The best individual that ever lived in a population, i.e. the *champion* can also be extracted as:

.. code-block:: c++

    // The decision vector
    pop1.champion_x()
    
    // The fitness vector
    pop1.champion_f()

Output:

.. code-block:: none

    { 1.0000000, 2.0000000, 3.0000000, 4.0000000 }
    { 8.4346944 }

.. note:: The *champion* is not necessarily identical to the best individual in the current population
          as it actually keeps the memory of all past individuals that were at some point in the population
