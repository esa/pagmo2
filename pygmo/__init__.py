# -*- coding: utf-8 -*-

from __future__ import absolute_import as _ai

__all__ = ['core']

# Problem extract functionality.
def _problem_extract(self,t):
    """Extract user-defined problem instance.

    If *t* is the same type of the user-defined problem used to construct this problem, then a deep copy of
    the user-defined problem will be returned. Otherwise, ``None`` will be returned.

    :param t: the type of the user-defined problem to extract
    :type t: ``type``
    :returns: a deep-copy of the internal user-defined problem if it is of type *t*, or ``None`` otherwise
    :rtype: *t*
    :raises: :exc:`TypeError` if *t* is not a type

    """
    if not isinstance(t,type):
        raise TypeError("the 't' parameter must be a type")
    if hasattr(t,"_pygmo_cpp_problem"):
        try:
            return self._cpp_extract(t())
        except TypeError:
            return None
    try:
        return self._py_extract(t)
    except TypeError:
        return None

def _problem_is(self,t):
    """Check the type of the user-defined problem instance.

    If *t* is the same type of the user-defined problem used to construct this problem, then ``True`` will be
    returned. Otherwise, ``False`` will be returned.

    :param t: the type of the user-defined problem to extract
    :type t: ``type``
    :returns: a boolean indicating whether the user-defined problem is of type *t* or not
    :rtype: ``bool``
    :raises: :exc:`TypeError` if *t* is not a type

    """
    return not self.extract(t) is None

# Algorithm extract functionality.
def _algorithm_extract(self,t):
    """Extract user-defined algorithm instance.

    If *t* is the same type of the user-defined algorithm used to construct this algorithm, then a deep copy of
    the user-defined algorithm will be returned. Otherwise, ``None`` will be returned.

    :param t: the type of the user-defined algorithm to extract
    :type t: ``type``
    :returns: a deep-copy of the internal user-defined algorithm if it is of type *t*, or ``None`` otherwise
    :rtype: *t*
    :raises: :exc:`TypeError` if *t* is not a type

    """
    if not isinstance(t,type):
        raise TypeError("the 't' parameter must be a type")
    if hasattr(t,"_pygmo_cpp_algorithm"):
        try:
            return self._cpp_extract(t())
        except TypeError:
            return None
    try:
        return self._py_extract(t)
    except TypeError:
        return None

def _algorithm_is(self,t):
    """Check the type of the user-defined algorithm instance.

    If *t* is the same type of the user-defined algorithm used to construct this algorithm, then ``True`` will be
    returned. Otherwise, ``False`` will be returned.

    :param t: the type of the user-defined algorithm to extract
    :type t: ``type``
    :returns: a boolean indicating whether the user-defined algorithm is of type *t* or not
    :rtype: ``bool``
    :raises: :exc:`TypeError` if *t* is not a type

    """
    return not self.extract(t) is None

from .core import *

# Override of the population constructor.
__original_population_init = population.__init__

def _population_init(self,prob=None,size=0,seed=None):
    """
    Constructor documentation:

    Args:
        prob: a user-defined problem (either Python or C++), or an instance of :class:`~pygmo.core.problem`
            (if ``None``, the population problem will be :class:`~pygmo.core.null_problem`)
        size (int): the number of individuals
        seed (int): the random seed (if ``None``, it will be randomly-generated)

    Raises:
        TypeError: if *size* is not an ``int`` or *seed* is not ``None`` and not an ``int``
        OverflowError:  is *size* or *seed* are negative
        unspecified: any exception thrown by the invoked C++ constructors or by the constructor of
            :class:`~pygmo.core.problem`

    """
    import sys
    if sys.version_info[0] < 3:
        int_types = (int,long)
    else:
        int_types = (int,)
    # Check input params.
    if not isinstance(size,int_types):
        raise TypeError("the 'size' parameter must be an integer")
    if not seed is None and not isinstance(size,int_types):
        raise TypeError("the 'seed' parameter must be None or an integer")
    if prob is None:
        # Use the null problem for default init.
        prob = null_problem()
    if type(prob) == problem:
        # If prob is a pygmo problem, we will pass it as-is to the
        # original init.
        prob_arg = prob
    else:
        # Otherwise, we attempt to create a problem from it. This will
        # work if prob is an exposed C++ problem or a Python UDP.
        prob_arg = problem(prob)
    if seed is None:
        __original_population_init(self,prob_arg,size)
    else:
        __original_population_init(self,prob_arg,size,seed)

setattr(population,"__init__",_population_init)

# Setup the extract and is methods for problem and meta-problems.
setattr(problem,"extract",_problem_extract)
setattr(problem,"is_",_problem_is)
setattr(translate,"extract",_problem_extract)
setattr(translate,"is_",_problem_is)

# Same for algorithm and meta-algorithms.
setattr(algorithm,"extract",_algorithm_extract)
setattr(algorithm,"is_",_algorithm_is)

# Register the cleanup function.
import atexit as _atexit
from .core import _cleanup
_atexit.register(lambda : _cleanup())
