# -*- coding: utf-8 -*-

from __future__ import absolute_import as _ai

__all__ = ['core']

# Problem extract functionality.
def _problem_extract(self,t):
    """Extract concrete problem instance.

    If *t* is the same type of the object *o* used to construct this problem, then a deep copy of
    *o* will be returned. Otherwise, ``None`` will be returned.

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

from .core import *

setattr(problem,"extract",_problem_extract)
setattr(translate,"extract",_problem_extract)
