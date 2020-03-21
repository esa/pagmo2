WFG problem test suite
======================================================

.. versionadded:: 2.11

.. cpp:namespace-push:: pagmo

.. cpp:class:: wfg

   This test suite was conceived to exceed the functionalities of previously implemented
   test suites. In particular, non-separable problems, deceptive problems, truly degenerative
   problems and mixed shape Pareto front problems are thorougly covered, as well as scalable problems
   in both the number of objectives and variables. Also, problems with dependencies between position and distance
   related parameters are covered.
   The WFG test suite was introduced by Simon Huband, Luigi Barone, Lyndon While, and Phil Hingston. In their paper the 
   authors identify the absence of nonseparable multimodal problems in order to test multi-objective optimization algorithms.
   In view of this, they propose a set of 9 different scalable multi-objective unconstrained problems (both in their
   objectives and in their decision vectors).
   All these problems, were developed satisfying the following guidelines:
   
   1) A few unimodal test problems should be present in the test suite. Various Pareto optimal geometries and bias
      conditions should define these problems, in order to test how the convergence velocity is influenced by these aspects.

   2) The following three Pareto optimal geometries should be present in the test suite: degenerate Pareto optimal
      fronts, disconnected Pareto optimal fronts and disconnected Pareto optimal sets.

   3) Many problems should be multimodal, and a few deceptive problems should also be covered.

   4) The majority of test problems should be non-separable.

   5) Both non-separable and multimodal problems should also be addressed.

   More in details:

   WFG1:

   This problems skews the relative significance of different parameters by employing different weights in the
   weighted sum reduction. Also, this problem is unimodal and with a convex and mixed Pareto optimal geometry.

   WFG2:

   This problem is non-separable, unimodal and with a convex and disconnected Pareto optimal geometry.

   WFG3:

   This is a non-separable, unimodal problem in all its objective except for the last one, which is multimodal.

   WFG4:

   This is a separable, multimodal problem with a concave Pareto optimal geometry. The multimodality
   of this problem has larger "hill sizes" than that of WFG9: this makes it thus more difficult.

   WFG5:

   This is a deceptive, separable problem with a concave Pareto optimal geometry.

   WFG6:

   This problem is non-separable and unimodal. Its Pareto optimal geometry is concave.The non-separable
   reduction of this problem makes it more difficult than that of WFG2 and WFG3.

   WFG7:

   This problem is separable, unimodal and with a concave Pareto optimal geometry. This, together with 
   WFG1, is the only problem that is both separable and unimodal.

   WFG8:

   This is a non-separable, unimodal problem with a concave Pareto optimal geometry.
 
   WFG9:
   
   This is a multimodal, deceptive and non-separable problem with a concave Pareto optimal geometry.
   Similar to WFG6, the non-separable reduction of this problem makes it more difficult than that of
   WFG2 and WFG3. Also, this problem is only deceptive on its position parameters.

   See: Huband, Simon, Hingston, Philip, Barone, Luigi and While Lyndon. "A Review of Multi-Objective Test Problems and a Scalable Test Problem Toolkit". IEEE Transactions on Evolutionary Computation (2006), 10(5), 477-506. doi: 10.1109/TEVC.2005.861417.
  
   .. cpp:function:: wfg(unsigned prob_id = 1u, vector_double::size_type dim_dvs = 5u, vector_double::size_type dim_obj = 3u, vector_double::size_type dim_k = 4u)

      Will construct one problem from the Walking Fish Group (WFG) test-suite..

      :param prob_id: problem number. Must be in [1, ..., 9].
      :param dim_dvs: decision vector dimension.
      :param dim_obj: objective function dimension.
      :param dim_k: position parameter. This parameter influences the shape functions of the various problems.
      :exception std\:\:invalid_argument: if *prob_id* is not in [1, ..., 9].
      :exception std\:\:invalid_argument: if *dim_dvs* is not >=1.
      :exception std\:\:invalid_argument: if *dim_obj* is not >=2.
      :exception std\:\:invalid_argument: if *dim_k* is not < *dim_dvs* , or is not >=1, or *dim_k* mod( *dim_obj* -1)!=0.
      :exception std\:\:invalid_argument: if *prob_id* =2 or *prob_id* =3 and ( *dim_dvs* - *dim_k* )mod(2)!=0. 

   .. cpp:function:: vector_double fitness(const vector_double &x) const

      Computes the fitness for this UDP.
    
      :param x: the decision vector.
      :return: the fitness of *x*.

   .. cpp:function:: std::pair<vector_double, vector_double> get_bounds() const

      Returns the box-bounds for this UDP.
     
      :return: the lower and upper bounds for each of the decision vector components.

   .. cpp:function:: std::string get_name() const

      Returns the problem name.

      :return: a string containing the problem name: "WFG *prob_id*".

   .. cpp:function:: template <typename Archive> void serialize(Archive &ar, unsigned)

      Object serialization.

      This method will save/load this into the archive *ar*.

      :param ar: target archive.
      :exception unspecified: any exception thrown by the serialization of primitive types.



