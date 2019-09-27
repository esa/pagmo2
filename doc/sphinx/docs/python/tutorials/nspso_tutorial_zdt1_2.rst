.. _py_tutorial_nspso_tutorial_zd1_2:

Benchmarking Nondominated Sorting Particle Swarm Optimizer (NSPSO) on ZDT1 and ZDT2 problems
=============================================================================================

In this tutorial we will benchmark the NSPSO algorithm, which can be found in the pygmo class :class:`~pygmo.nspso`, on the first two problems of the Zitzler, Deb and Thiele (ZDT) test problem suite (i.e., ZDT1 and ZDT2).

In particular, in an attempt of reproducing a similar strategy as the one proposed in the original paper where NSPSO was introduced (i.e., "A Non-dominated Sorting Particle Swarm Optimizer for Multiobjective Optimization" by Xiadong Li), we check the values of two performance metrics for three different algorithms (i.e., NSPSO with crowding distance, NSPSO with niche count and NSGA-II) on ZDT1 and ZDT2. In particular, while in the original paper the Delta and the GD metrics were used, in this case we use the average of the crowding distance as diversity metric and the p-distance as convergence metric. Both these metrics are already available for the use in pagmo.

The crowding distance can be computed in any population set, by simply calling the class :class:`~pygmo.crowding_distance`, giving as input the relevant population where the crowding distance wants to be computed for the non dominated front. On the other hand, the p-distance value is only available for the ZDT and DTLZ test suites and it can be retrieved by calling the *p_distance* method of the :class:`~pygmo.zdt` class, passing as input the population set. 

Both ZDT1 and ZDT2 problems are box-constrained continuous bi-objectives problem and are provided as UDP (user-defined problem) by pygmo in the :class:`~pygmo.zdt` class. 
For benchmarking the aforesaid algorithms, we will use the same input parameters suggested in the original paper. Thus, for the two NSPSO algorithms, we will use the same set of input parameters, but using two different diversity strategies (i.e., crowding distance and niche count). In the first case, we will thus have: *algo = algorithm(nspso(gen = 100, omega=0.001, c1 = 2.0, c2 = 2.0, chi = 1.0, v_coeff = 0.5, leader_selection_range = 100, diversity_mechanism = 'crowding distance', memory = False, seed = 20))*, whereas in the second: *algo = algorithm(nspso(gen = 100, omega=0.001, c1 = 2.0, c2 = 2.0, chi = 1.0, v_coeff = 0.5, leader_selection_range = 100, diversity_mechanism = 'niche count', memory = False, seed = 20))*. Similarly, for NSGA-II, we will also use the same input parameters also recommended in the original paper: *algo_3 = algorithm(nsga2(gen = 100, cr=0.9, m=1/30, eta_c=20, eta_m=20, seed = 20))*.
Furthermore, the same population and generation sizes will be chosen: 200 and 100, respectively. 

It has to be noted that our purpose is not to exactly reproduce the original paper's results: indeed, pagmo's version of NSPSO is slightly different. First of all, the inertia weight parameter (i.e., *omega*) is not varied dynamically, but the user can decide to implement a schedule and vary it on a generation by generation basis. Indeed, thanks to the *memory* parameter, NSPSO can be called either in one single call or iteratively in a for loop, by maintaining the same results. This allows the users to access the algorithm's population generation by generation and to possibly change the input parameters. Secondly, in pagmo we have introduced the leader selection range parameter (i.e., *leader_selection_range*), which was not present in the original implementation.

Each of the UDAs is run ten times on each problem and the results are eventually averaged and shown in terms of mean and standard deviation. The following piece of code is used for performing the benchmark:

.. doctest::
 
    >>> from pygmo import *
    >>> from pygmo import *
    >>> import numpy as np
    >>> pop_size=200
    >>> problems=[1,2]
    >>> #We declare empty arrays for storing the p-distance and mean crowding distance values for all
    >>> #the algorithms and problems over 10 runs:
    >>> #p-distance
    >>> p_dist_nspso_cd_zdt1=[]
    >>> p_dist_nspso_cd_zdt2=[]
    >>> p_dist_nspso_nc_zdt1=[]
    >>> p_dist_nspso_nc_zdt2=[]
    >>> p_dist_nsga2_zdt1=[]
    >>> p_dist_nsga2_zdt2=[]
    >>> #crowding distance
    >>> mean_cd_nspso_cd_zdt1=[]
    >>> mean_cd_nspso_cd_zdt2=[]
    >>> mean_cd_nspso_nc_zdt1=[]
    >>> mean_cd_nspso_nc_zdt2=[]
    >>> mean_cd_nsga2_zdt1=[]
    >>> mean_cd_nsga2_zdt2=[]

    >>> #We run the algos ten times each, and we store p-distance and crowding distance
    >>> for j in problems:
    ...       # 1. We declare the problem (either ZDT1 or ZDT2):
    ...       if j==1:
    ...           udp=zdt(prob_id=1)
    ...       elif j==2:
    ...           udp=zdt(prob_id=2)
    ...                   
    ...       for ii in range(0,10):
    ...             # 2. We declare the three populations to be evolved:
    ...             pop_1 = population(prob = udp, size = pop_size, seed = ii+3)
    ...             pop_2 = population(prob = udp, size = pop_size, seed = ii+3)
    ...             pop_3 = population(prob = udp, size = pop_size, seed = ii+3)
    ...             # 3. We declare the algorithms to be used: NSPSO with crowding distance, NSPSO with niche count and NSGA-II:
    ...             algo = algorithm(nspso(gen = 100, omega=0.001, c1 = 2.0, c2 = 2.0, chi = 1.0, v_coeff = 0.5, leader_selection_range = 100, diversity_mechanism = 'crowding distance', memory = False, seed = 20))
    ...             algo_2 = algorithm(nspso(gen = 100, omega=0.001, c1 = 2.0, c2 = 2.0, chi = 1.0, v_coeff = 0.5, leader_selection_range = 100, diversity_mechanism = 'niche count', memory = False, seed = 20)) 
    ...             algo_3 = algorithm(nsga2(gen = 100, cr=0.9, m=1/30, eta_c=20, eta_m=20, seed = 20))
    ...             # 4. We evolve the populations for the three algorithms:
    ...             pop_1 = algo.evolve(pop_1)
    ...             pop_2 = algo_2.evolve(pop_2)
    ...             pop_3 = algo_3.evolve(pop_3)
    ...             
    ...             #This returns the first (i.e., best) non-dominated front:
    ...             nds_nspso_cd = non_dominated_front_2d(pop_1.get_f())
    ...             nds_nspso_nc = non_dominated_front_2d(pop_2.get_f())
    ...             nds_nsga2    = non_dominated_front_2d(pop_3.get_f())
    ...             
    ...             #We store all the non-dominated fronts crowding distances, for all the algorithms:
    ...             cd_nspso_cd  = crowding_distance(pop_1.get_f()[nds_nspso_cd])
    ...             cd_nspso_nc  = crowding_distance(pop_2.get_f()[nds_nspso_nc])
    ...             cd_nsga2     = crowding_distance(pop_3.get_f()[nds_nsga2])
    ...             
    ...             # 5. We compute the p-dist and store it in a vector, for each problem and each algorithm:
    ...             if j==1: #ZDT1
    ...                 #We gather the crowding distance means:
    ...                 mean_cd_nspso_cd_zdt1.append(np.mean(cd_nspso_cd[np.isfinite(cd_nspso_cd)]))
    ...                 mean_cd_nspso_nc_zdt1.append(np.mean(cd_nspso_cd[np.isfinite(cd_nspso_cd)]))
    ...                 mean_cd_nsga2_zdt1.append(np.mean(cd_nsga2[np.isfinite(cd_nsga2)]))
    ...                 #And the p-distance values:
    ...                 p_dist_nspso_cd_zdt1.append(udp.p_distance(pop_1))
    ...                 p_dist_nspso_nc_zdt1.append(udp.p_distance(pop_2))
    ...                 p_dist_nsga2_zdt1.append(udp.p_distance(pop_3))
    ...             elif j==2: #ZDT2
    ...                 #We gather the crowding distance means:
    ...                 mean_cd_nspso_cd_zdt2.append(np.mean(cd_nspso_cd[np.isfinite(cd_nspso_cd)]))
    ...                 mean_cd_nspso_nc_zdt2.append(np.mean(cd_nspso_cd[np.isfinite(cd_nspso_cd)]))
    ...                 mean_cd_nsga2_zdt2.append(np.mean(cd_nsga2[np.isfinite(cd_nsga2)]))
    ...                 #And the p-distance values:
    ...                 p_dist_nspso_cd_zdt2.append(udp.p_distance(pop_1))
    ...                 p_dist_nspso_nc_zdt2.append(udp.p_distance(pop_2))
    ...                 p_dist_nsga2_zdt2.append(udp.p_distance(pop_3))

Once that we have run the three algorithms on the ZDT1 and ZDT3 problems, by storing all the crowding distances and p-distances values, we can show the results:

.. doctest::

    >>> # 6. We print the results: #doctest: +SKIP 
    >>> print("\n NSPSO with crowding distance:") #doctest: +SKIP 
    >>> print("ZDT1-> p-distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(p_dist_nspso_cd_zdt1), "std":np.nanstd(p_dist_nspso_cd_zdt1)}) #doctest: +SKIP 
    >>> print("ZDT2-> p-distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(p_dist_nspso_cd_zdt2), "std":np.nanstd(p_dist_nspso_cd_zdt2)}) #doctest: +SKIP 
    >>> print("ZDT1-> crowding distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(mean_cd_nspso_cd_zdt1), "std":np.nanstd(mean_cd_nspso_cd_zdt1)}) #doctest: +SKIP 
    >>> print("ZDT2-> crowding distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(mean_cd_nspso_cd_zdt2), "std":np.nanstd(mean_cd_nspso_cd_zdt2)}) #doctest: +SKIP 
    <BLANKLINE>
    >>> print("\n NSPSO with niche count:") #doctest: +SKIP 
    >>> print("ZDT1-> p-distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(p_dist_nspso_nc_zdt1), "std":np.nanstd(p_dist_nspso_nc_zdt1)}) #doctest: +SKIP 
    >>> print("ZDT2-> p-distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(p_dist_nspso_nc_zdt2), "std":np.nanstd(p_dist_nspso_nc_zdt2)}) #doctest: +SKIP 
    >>> print("ZDT1-> crowding distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(mean_cd_nspso_cd_zdt1), "std":np.nanstd(mean_cd_nspso_cd_zdt1)}) #doctest: +SKIP 
    >>> print("ZDT2-> crowding distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(mean_cd_nspso_cd_zdt2), "std":np.nanstd(mean_cd_nspso_cd_zdt2)}) #doctest: +SKIP 
    <BLANKLINE>
    >>> print("\n NSGA2:") #doctest: +SKIP 
    >>> print("ZDT1-> p-distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(p_dist_nsga2_zdt1), "std":np.nanstd(p_dist_nsga2_zdt1)}) #doctest: +SKIP 
    >>> print("ZDT2-> p-distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(p_dist_nsga2_zdt2), "std":np.nanstd(p_dist_nsga2_zdt2)}) #doctest: +SKIP 
    >>> print("ZDT1-> crowding distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(mean_cd_nsga2_zdt1), "std":np.nanstd(mean_cd_nsga2_zdt1)}) #doctest: +SKIP 
    >>> print("ZDT2-> crowding distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(mean_cd_nsga2_zdt2), "std":np.nanstd(mean_cd_nsga2_zdt2)}) #doctest: +SKIP 
     NSPSO with crowding distance: #doctest: +SKIP 
     ZDT1-> p-distance mean and std: 0.054309 +/- 0.028563 #doctest: +SKIP 
     ZDT2-> p-distance mean and std: 0.020207 +/- 0.016466 #doctest: +SKIP 
     ZDT1-> crowding distance mean and std: 0.020099 +/- 0.000038 #doctest: +SKIP 
     ZDT2-> crowding distance mean and std: 0.020065 +/- 0.000102 #doctest: +SKIP 
    <BLANKLINE>
     NSPSO with niche count: #doctest: +SKIP 
     ZDT1-> p-distance mean and std: 0.054797 +/- 0.016863 #doctest: +SKIP
     ZDT2-> p-distance mean and std: 0.011945 +/- 0.010522 #doctest: +SKIP
     ZDT1-> crowding distance mean and std: 0.049834 +/- 0.009898 #doctest: +SKIP
     ZDT2-> crowding distance mean and std: 0.044450 +/- 0.010312 #doctest: +SKIP
    <BLANKLINE>
     NSGA2:
     ZDT1-> p-distance mean and std: 0.011525 +/- 0.001534 #doctest: +SKIP
     ZDT2-> p-distance mean and std: 0.009290 +/- 0.001335 #doctest: +SKIP
     ZDT1-> crowding distance mean and std: 0.020099 +/- 0.000038 #doctest: +SKIP
     ZDT2-> crowding distance mean and std: 0.020065 +/- 0.000102 #doctest: +SKIP
