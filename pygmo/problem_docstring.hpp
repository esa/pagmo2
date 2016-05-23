#ifndef PYGMO_PROBLEM_DOCSTRING_HPP
#define PYGMO_PROBLEM_DOCSTRING_HPP

namespace pygmo
{

constexpr auto problem_docstring = R"(The main problem class.

>>> from pygmo import problem, rosenbrock
>>> p = problem(rosenbrock(dim=5)) # Construction from an exposed C++ problem.
>>> p # doctest: +SKIP
Problem name: Multidimensional Rosenbrock Function
        Global dimension:                       5
        Fitness dimension:                      1
        Number of objectives:                   1
        Equality constraints dimension:         0
        Inequality constraints dimension:       0
        Lower bounds: [-5, -5, -5, -5, -5]
        Upper bounds: [1, 1, 1, 1, 1]
<BLANKLINE>
        Has gradient: false
        User implemented gradient sparsity: false
        Has hessians: false
        User implemented hessians sparsity: false
<BLANKLINE>
        Function evaluations: 0
<BLANKLINE>
>>> p.fitness([1,2,3,4,5])
array([ 14814.])
>>> class prob: # Definition of a pure-Python problem.
...   def fitness(self,dv):
...     return [dv[0]*dv[0]]
...   def get_bounds(self):
...     return ([1],[2])
...   def get_name(self):
...     return "A simple problem"
>>> p = problem(prob()) # Construction from the user-defined Python problem.
>>> p # doctest: +SKIP
Problem name: A simple problem
        Global dimension:                       1
        Fitness dimension:                      1
        Number of objectives:                   1
        Equality constraints dimension:         0
        Inequality constraints dimension:       0
        Lower bounds: [1]
        Upper bounds: [2]
<BLANKLINE>
        Has gradient: false
        User implemented gradient sparsity: false
        Has hessians: false
        User implemented hessians sparsity: false
<BLANKLINE>
        Function evaluations: 0
<BLANKLINE>
>>> p.fitness([2])
array([ 4.])

)";

}

#endif
