.. _cpp_tutorial_rewriting_decoders:

Rewriting decision vectors during the fitness evaluation
========================================================

.. versionadded:: 2.21.0

In some optimisation approaches, the fitness evaluation does more than just
computing objectives and constraints: it also *improves* the solution being evaluated,
and the improved solution should replace the original one. A prominent example are
the biased random-key genetic algorithms (BRKGA), where a *decoder* maps a vector of
random keys (the chromosome) to a solution, possibly improves the solution with a local
search, and then *rewrites* the chromosome so that it encodes the improved solution.
Writing the result of the local search back into the chromosome turns a memetic algorithm
from a Baldwinian one (only the fitness benefits from the local search) into a Lamarckian
one (the improved chromosomes are inherited by the offspring).

In pagmo, a user-defined problem (UDP) can opt in to this behaviour by implementing,
in addition to the mandatory ``fitness()`` member function, the optional member function

.. code-block:: c++

   vector_double fitness_rewrite(vector_double &dv) const;

which must return the fitness of the decision vector *dv* **after** having rewritten it in place.
The UDP can also implement

.. code-block:: c++

   bool has_fitness_rewrite() const;

in order to enable or disable the rewriting at runtime (see :cpp:func:`pagmo::problem::has_fitness_rewrite()`).

The rewritten decision vector must have the same size as the original one, it must lie within the problem
bounds and its integer part (if any) must contain integral values: pagmo checks these conditions and raises an
error otherwise, leaving the original decision vector untouched. Ideally, rewriting should also be idempotent
(i.e., rewriting an already rewritten decision vector should not change it), so that each stored decision vector
is a fixed point of the decoder.

The ``fitness()`` member function must not modify its argument. A convenient way of implementing both member
functions, used in the example below and borrowed from the
`BRKGA-MP-IPR <https://github.com/ceandrade/brkga_mp_ipr_cpp>`__ library, is to delegate them to a single
``decode(dv, rewrite)`` function.

How rewriting is used by pagmo
------------------------------

Rewriting happens through :cpp:func:`pagmo::problem::fitness_rewrite()` (and its batch counterpart,
:cpp:func:`pagmo::bfe::fitness_rewrite()`). If the UDP does not support rewriting, these functions
behave exactly like :cpp:func:`pagmo::problem::fitness()` (and the call operator of :cpp:class:`pagmo::bfe`),
so that existing problems are not affected in any way.

The :cpp:class:`~pagmo::population` class stores the rewritten decision vectors whenever it evaluates
the fitness itself, that is, in its constructors, in the single-argument overloads of
:cpp:func:`~pagmo::population::push_back()` and in :cpp:func:`~pagmo::population::set_x()`. The
overloads which receive both a decision vector and a fitness vector (such as
:cpp:func:`~pagmo::population::set_xf()`) store their arguments as they are.

The following algorithms store the rewritten decision vectors they generate, both when evaluating
serially and when using a batch fitness evaluator:

* multi-objective: :cpp:class:`~pagmo::nsga2`, :cpp:class:`~pagmo::nsga3`, :cpp:class:`~pagmo::moead`,
  :cpp:class:`~pagmo::moead_gen`, :cpp:class:`~pagmo::nspso`, :cpp:class:`~pagmo::maco`;
* single-objective genetic and evolutionary algorithms: :cpp:class:`~pagmo::sga`, :cpp:class:`~pagmo::sea`,
  :cpp:class:`~pagmo::de`, :cpp:class:`~pagmo::de1220`, :cpp:class:`~pagmo::sade`;
* swarm and other heuristics: :cpp:class:`~pagmo::pso`, :cpp:class:`~pagmo::pso_gen`, :cpp:class:`~pagmo::gwo`,
  :cpp:class:`~pagmo::bee_colony`, :cpp:class:`~pagmo::ihs`, :cpp:class:`~pagmo::gaco`,
  :cpp:class:`~pagmo::simulated_annealing`, :cpp:class:`~pagmo::compass_search`;
* local optimisers: :cpp:class:`~pagmo::nlopt` and :cpp:class:`~pagmo::ipopt` rewrite the final solution
  they find (but not the points they evaluate during the optimisation), before comparing it with the
  selected individual;
* meta-algorithms: :cpp:class:`~pagmo::cstrs_self_adaptive` forwards the rewriting to its inner algorithm
  (through the penalized problem it evolves), and :cpp:class:`~pagmo::mbh` rewrites the perturbed decision vectors
  (via :cpp:func:`~pagmo::population::set_x()`) as well as whatever its inner algorithm rewrites.

:cpp:class:`~pagmo::cmaes` and :cpp:class:`~pagmo::xnes`, on the other hand, evaluate the fitness via
:cpp:func:`pagmo::problem::fitness()`, and thus never rewrite the decision vectors they sample: their search
distribution update assumes that the evaluated decision vectors are the sampled ones. In all cases,
the stored decision vectors remain consistent with the stored fitness vectors.

Among the batch fitness evaluators provided by pagmo, :cpp:class:`~pagmo::thread_bfe` and
:cpp:class:`~pagmo::default_bfe` rewrite the decision vectors in parallel. Batch fitness evaluators which do not
support rewriting (such as :cpp:class:`~pagmo::member_bfe`, since the UDP's ``batch_fitness()`` member function
cannot modify its input) raise an error when used with a problem that rewrites decision vectors.

.. note::

   The fitness evaluation of a rewriting UDP is subject to the same thread safety requirements as
   the ``fitness()`` member function (see :cpp:func:`pagmo::problem::get_thread_safety()`).

.. note::

   The meta-problems provided by pagmo (e.g., :cpp:class:`~pagmo::decompose`, :cpp:class:`~pagmo::unconstrain`
   and :cpp:class:`~pagmo::translate`) do not forward rewriting to the problem they wrap: wrapping a rewriting UDP
   in a meta-problem results in plain fitness evaluations.

An example: a random-key bi-objective TSP
-----------------------------------------

In the following example, a bi-objective travelling salesman problem is encoded with random keys:
the decoder sorts the keys to obtain a tour, improves the tour with a 2-opt local search accepting
only the moves which improve one objective without worsening the other, and finally rewrites the keys so
that they encode the improved tour (by reassigning the sorted keys following the improved tour). The problem
is then solved with :cpp:class:`~pagmo::nsga2`, using a :cpp:class:`~pagmo::thread_bfe` to decode the
chromosomes in parallel.

.. literalinclude:: ../../../../../tutorials/rewriting_decoder.cpp
   :language: c++
   :linenos:

At the end of the evolution, every chromosome in the population is a fixed point of the decoder
(it encodes a 2-opt local optimum) and is consistent with its fitness.
