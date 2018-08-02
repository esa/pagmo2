.. _problems:

List of problems (UDPs) available in pagmo/pygmo
================================================

This is the list of user defined problems (UDPs) currently provided with pagmo/pygmo. These are classes that 
can be used to construct a :cpp:class:`pagmo::problem`, or :class:`pygmo.problem` which will then provide a unified 
interface to acces their functionalities.

Each problem can be can be of type S = Single, M = Multi-objective, I = has an integer part, C = Constrained, U = Unconstrained, sto = stochastic

Scalable problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
========================================================== ========================================= ========================================= =============== 
Common Name                                                Docs of the C++ class                     Docs of the python class                  Type            
========================================================== ========================================= ========================================= =============== 
Ackley                                                     :cpp:class:`pagmo::ackley`                :class:`pygmo.ackley`                     S-U
Griewank                                                   :cpp:class:`pagmo::griewank`              :class:`pygmo.griewank`                   S-U
Hock Schittkowsky 71                                       :cpp:class:`pagmo::hock_shittkowski_71`   :class:`pygmo.hock_shittkowski_71`        S-C
Inventory                                                  :cpp:class:`pagmo::inventory`             :class:`pygmo.inventory`                  S-U (sto)
Luksan Vlcek 1                                             :cpp:class:`pagmo::luksan_vlcek1`         :class:`pygmo.luksan_vlcek1`              S-C
Rastrigin                                                  :cpp:class:`pagmo::rastrigin`             :class:`pygmo.rastrigin`                  S-U
MINLP Rastrigin                                            :cpp:class:`pagmo::minlp_rastrigin`       :class:`pygmo.minlp_rastrigin`            S-U-I
Rosenbrock                                                 :cpp:class:`pagmo::rosenbrock`            :class:`pygmo.rosenbrock`                 S-U
Schwefel                                                   :cpp:class:`pagmo::schwefel`              :class:`pygmo.schwefel`                   S-U
========================================================== ========================================= ========================================= ===============

Problem suites 
^^^^^^^^^^^^^^^
================================== ============================================ ============================================ =============== 
Common Name                        Docs of the C++ class                        Docs of the python class                     Type            
================================== ============================================ ============================================ =============== 
CEC2006                            :cpp:class:`pagmo::cec2006`                  :class:`pygmo.cec2006`                       S-C           
CEC2009                            :cpp:class:`pagmo::cec2009`                  :class:`pygmo.cec2009`                       S-C           
CEC2013                            :cpp:class:`pagmo::cec2013`                  :class:`pygmo.cec2013`                       S-U           
CEC2014                            :cpp:class:`pagmo::cec2014`                  :class:`pygmo.cec2014`                       S-U           
ZDT                                :cpp:class:`pagmo::zdt`                      :class:`pygmo.zdt`                           M-U
DTLZ                               :cpp:class:`pagmo::dtlz`                     :class:`pygmo.dtlz`                          M-U
================================== ============================================ ============================================ =============== 

Meta-problems
^^^^^^^^^^^^^
========================================================== ========================================= =========================================
Common Name                                                Docs of the C++ class                     Docs of the python class                 
========================================================== ========================================= =========================================
Decompose                                                  :cpp:class:`pagmo::decompose`             :class:`pygmo.decompose`                 
Translate                                                  :cpp:class:`pagmo::translate`             :class:`pygmo.translate`                 
Unconstrain                                                :cpp:class:`pagmo::unconstrain`           :class:`pygmo.unconstrain`               
Decorator                                                  N/A                                       :class:`pygmo.decorator_problem`
========================================================== ========================================= =========================================
