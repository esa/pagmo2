Genetic Operators
======================================================

Some utilities to perform common genetic operations

*#include <pagmo/utils/genetic_operators.hpp>*

.. cpp:namespace-push:: pagmo

.. versionadded:: 2.16

.. cpp:function:: std::pair<vector_double, vector_double> sbx_crossover(const vector_double &parent1, const vector_double &parent2, const std::pair<vector_double, vector_double> &bounds, vector_double::size_type nix, const double p_cr, const double eta_c, detail::random_engine_type &random_engine)

   :param parent1: first parent.
   :param parent2: second parent.
   :param bounds: problem bounds.
   :param nix: integer dimension of the problem.
   :param p_cr: crossover probability.
   :param eta_c: crossover distribution index.
   :param random_engine: the pagmo random engine

   :exception std\:\:invalid_argument: if the *bounds* size is zero.
   :exception std\:\:invalid_argument: if *parent1*, *parent2* and *bounds* have unequal length.
   :exception std\:\:invalid_argument: if *bounds* contain any nan or infs.
   :exception std\:\:invalid_argument: if any lower bound is greater than the corresponding upper bound.
   :exception std\:\:invalid_argument: if *nix* is larger than bounds size.
   :exception std\:\:invalid_argument: if any of the integer bounds are not not integers.
   :exception std\:\:invalid_argument: if *p_cr* or *eta_c* are not finite numbers.


   This function perform a simulated binary crossover (SBX) among two parents.
   The SBX genetic operator was designed to preserve average property and the
   spread factor property of one-point crossover in binary encoded chromosomes.
   This version of the SBX will act as a simple two-points crossover over the
   integer part of the chromosome / decision vector.
  
   See: https://www.slideshare.net/paskorn/simulated-binary-crossover-presentation

.. versionadded:: 2.16

.. cpp:function:: void polynomial_mutation(vector_double &dv, const std::pair<vector_double, vector_double> &bounds, vector_double::size_type nix, const double p_m, const double eta_m, detail::random_engine_type &random_engine)

   :param dv: decision vector to be mutated in place.
   :param bounds: problem bounds.
   :param nix: integer dimension of the problem.
   :param m_cr: mutation probability.
   :param m_c: mutation distribution index.
   :param random_engine: the pagmo random engine

   :exception std\:\:invalid_argument: if the *bounds* size is zero.
   :exception std\:\:invalid_argument: if *dv*, and *bounds* have unequal length.
   :exception std\:\:invalid_argument: if *bounds* contain any nan or infs.
   :exception std\:\:invalid_argument: if any lower bound is greater than the corresponding upper bound.
   :exception std\:\:invalid_argument: if *nix* is larger than bounds size.
   :exception std\:\:invalid_argument: if any of the integer bounds are not not integers.
   :exception std\:\:invalid_argument: if *m_cr* or *m_c* are not finite numbers.


   This function performs the polynomial mutation proposed by Agrawal and Deb over some chromosome / decision vector.
  
   See: https://www.iitk.ac.in/kangal/papers/k2012016.pdf

   



