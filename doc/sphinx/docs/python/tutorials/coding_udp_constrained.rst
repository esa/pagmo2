.. _py_tutorial_coding_udp_constrained:

Coding a User Defined Problem with constraints
----------------------------------------------

We here show how to code a non-trivial user defined problem (UDP) with a single objective, equality and inequality constraints.
We assume that the mathematical formulation of problem is the following:

.. math::

   \begin{array}{rl}
   \mbox{minimize:} & \sum_{i=1}^3 \left[(x_{2i-1}-3)^2 / 1000 - (x_{2i-1}-x_{2i}) + \exp(20(x_{2i-1}-x_{2i}))\right]\\
   \mbox{subject to:} & \\
   & 4(x_1-x_2)^2+x_2-x_3^2+x_3-x_4^2  = 0 \\
   & 8x_2(x_2^2-x_1)-2(1-x_2)+4(x_2-x_3)^2+x_1^2+x_3-x_4^2+x_4-x_5^2 = 0 \\
   & 8x_3(x_3^2-x_2)-2(1-x_3)+4(x_3-x_4)^2+x_2^2-x_1+x_4-x_5^2+x_1^2+x_5-x_6^2 = 0 \\
   & 8x_4(x_4^2-x_3)-2(1-x_4)+4(x_4-x_5)^2+x_3^2-x_2+x_5-x_6^2+x_2^2+x_6-x_1 <= 0 \\
   & 8x_5(x_5^2-x_4)-2(1-x_5)+4(x_5-x_6)^2+x_4^2-x_3+x_6+x_3^2-x_2 >= 0 \\
   & 8x_6(x_6^2-x_5)-2(1-x_6)             +x_5^2-x_4+x_4^2-x_5 >= 0 \\
   \end{array}

which is a modified instance of the problem 5.9 in Luksan, L., and Jan Vlcek. "Sparse and partially separable test problems
for unconstrained and equality constrained optimization." (1999). The modification is in the last three constraints that are,
for the purpose of this tutorial, considered as inequalities rather than equality constraints.