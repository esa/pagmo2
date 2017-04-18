# -*- coding: iso-8859-1 -*-

try:
  from numpy import get_include
  print(get_include())
except ImportError:
  print('')
