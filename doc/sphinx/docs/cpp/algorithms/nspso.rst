Non dominated sorting particle swarm optimization(NSPSO)
===========================================================

.. versionadded:: 2.12

.. cpp:namespace-push:: pagmo

.. cpp:class:: nspso

   Non-dominated Sorting Particle Swarm Optimizer (NSPSO) is a modified version of PSO for multi-objective optimization.
   It extends the basic ideas of PSO by making a better use of personal bests and offspring for non-dominated comparison. In order to increase the diversity of the Pareto front it is possible to choose between 3 different niching methods: crowding distance, niche count and maxmin.|
   
   * See: Xiadong, Li. "A Non-dominated Sorting Particle Swarm Optimizer for Multiobjective Optimization". Genetic and Evolutionary Computation - GECCO (2003), vol. 2723, pp. 37-48, doi: https://doi.org/10.1007/3-540-45105-6_4.
   
   * See: Xiadong, Li. "Better Spread and Convergence: Particle Swarm Multiobjective Optimization Using the Maximin Fitness Function". Genetic and Evolutionary Computation - GECCO (2004), vol. 3102, pp. 117-128, doi: https://doi.org/10.1007/978-3-540-24854-5_11.
   
   * See: Fonseca, M. Carlos and Fleming, J. Peter. "Genetic Algorithms for Multiobjective Optimization: Formulation, Discussion and Generalization". Proceedings of the ICGA-93: Fifth International Conference on Genetic Algorithms (1993), vol. 3, pp. 416-423.

   This constructor will construct NSPSO.

   .. cpp:function:: nspso(unsigned gen = 1u, double min_w = 0.95, double max_w = 10., double c1 = 0.01, double c2 = 0.5, double chi = 0.5, double v_coeff = 0.5, unsigned leader_selection_range = 2u, std::string diversity_mechansim = "crowding distance", unsigned seed = pagmo::random_device::next())

      :param `gen`: number of generations to evolve.
      :param `omega`: particles' inertia weight.
      :param `c1`: magnitude of the force, applied to the particle's velocity, in the direction of its previous best position.
      :param `c2`: magnitude of the force, applied to the particle's velocity, in the direction of its global best (i.e., leader).
      :param `chi`: velocity scaling factor.
      :param `v_coeff`: velocity coefficient (determining the maximum allowed particle velocity).
      :param `leader_selection_range`: leader selection range parameter (i.e., the leader of each particle is selected among the best `leader_selection_range` % `individuals`).
      :param `diversity_mechanism`: the diversity mechanism used to maintain diversity on the Pareto front.
      :param `memory`: memory parameter. If `true`, memory is activated in the algorithm for multiple calls.
      :param `seed`: seed used by the internal random number generator (default is random).
      :exception `std\:\:invalid_argument`: if  `c1` <= 0, or `c2` <= 0, or `chi` <= 0.
      :exception `std\:\:invalid_argument`: if `omega` < 0, or `omega` > 1,.
      :exception `std\:\:invalid_argument`: if `v_coeff` <= 0, or `v_coeff` > 1.
      :exception `std\:\:invalid_argument`: if `leader_selection_range` > 100.
      :exception `std\:\:invalid_argument`: if `diversity_mechanism` is not "*crowding distance*", or "*niche count*", or "*max min*".

   .. cpp:function:: population evolve(population pop) const

      Algorithm evolve method: evolves the population for the requested number of generations.
    
      :param pop: population to be evolved.
      :return: evolved population.
      :throw: ``std::invalid_argument`` if ``pop.get_problem()`` is stochastic, single objective or has non linear constraints. If the population size is smaller than 2.

   .. cpp:function:: void set_seed(unsigned seed)

      Sets the seed.
      
      :param ``seed``: the seed controlling the algorithm stochastic behaviour.

   .. cpp:function:: unsigned get_seed(unsigned seed)

      Gets the seed.
      
      :return: the seed controlling the algorithm stochastic behaviour.

   .. cpp:function:: void set_verbosity(unsigned level)

      Sets the algorithm verbosity: sets the verbosity level of the screen ouput and of the log returned by ``get_log()`. *level* can be: 
      - 0: no verbosity.
      - >0: will print and log one line each *level* generations.
      Example (verbosity 1, where Gen, is the generation number, Fevals the number of function evaluations used; also, the ideal point of the current population follows cropped to its 5th component):
   .. code-block:: c++
      :linenos:

      Gen:        Fevals:        ideal1:        ideal2:        ideal3:        ideal4:        ideal5:          ... :
         1             52      0.0586347      0.0587097      0.0586892      0.0592426      0.0614239
         2            104     0.00899252     0.00899395     0.00945782      0.0106282      0.0276778
         3            156     0.00899252     0.00899395     0.00945782      0.0106282      0.0276778
         4            208     0.00899252     0.00899395     0.00945782      0.0106282      0.0276778
         5            260     0.00899252     0.00899395     0.00945782      0.0106282      0.0276778
         6            312     0.00899252     0.00899395     0.00945782      0.0106282      0.0276778
         7            364     0.00899252     0.00899395     0.00945782      0.0106282      0.0276778
         8            416     0.00899252     0.00899395     0.00945782      0.0106282      0.0276778
         9            468     0.00899252     0.00899395     0.00945782      0.0106282      0.0276778
        10            520     0.00899252     0.00899395     0.00945782      0.0106282      0.0276778

   .. cpp:function:: unsigned get_verbosity() const

      Gets the verbosity level.
      
      :return: the verbosity level.

   .. cpp:function:: unsigned get_gen() const

      Gets the generations.
      
      :return: the number of generations to evolve for.

   .. cpp:function:: const log_type &get_log() const

      Gets the log. A log containing relevant quantities monitoring the last call to evolve. Each element of the returned ``std::vector`` is a ``nspso::log_line_type`` containing: Gen, Fevals, ideal_point as described in ``nspso::set_verbosity``.
      
      :return: an ``std::vector`` of ``nspso::log_line_type`` containing the logged values Gen, Fevals, ideal_point.

   .. cpp:function:: void set_bfe(const bfe &b)

      Sets the batch function evaluation scheme.
      
      :param ``b``: batch function evaluation object.

   .. cpp:function:: std::string get_extra_info() const

      Extra info. Returns extra information on the algorithm.
      
      :return: an ``std::string`` containing extra info on the algorithm.
     
   .. cpp:function:: std::string get_name() const

      Returns the problem name.

      :return: a string containing the problem name: "NSPSO".

   .. cpp:function:: template <typename Archive> void serialize(Archive &ar, unsigned)

      Object serialization.

      This method will save/load this into the archive *ar*.

      :param ``ar``: target archive.
      :exception unspecified: unspecified any exception thrown by the serialization of the UDA and of primitive types.



