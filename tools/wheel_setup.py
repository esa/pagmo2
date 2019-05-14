from setuptools import setup
from setuptools.dist import Distribution
import sys

NAME = 'pygmo'
VERSION = '@pygmo_VERSION@'
DESCRIPTION = 'Parallel optimisation for C++ / Python'
LONG_DESCRIPTION = 'A platform to perform parallel computations of optimisation tasks (global and local) via the asynchronous generalized island model.'
URL = 'https://github.com/esa/pagmo2'
AUTHOR = 'The pagmo development team'
AUTHOR_EMAIL = 'pagmo@googlegroups.com'
LICENSE = 'GPLv3+/LGPL3+'
CLASSIFIERS = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 4 - Beta',

    'Operating System :: OS Independent',

    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',

    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',

    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3'
]
KEYWORDS = 'science math physics optimization ai evolutionary-computing parallel-computing metaheuristics'
INSTALL_REQUIRES = ['numpy', 'cloudpickle']
PLATFORMS = ['Unix', 'Windows', 'OSX']


class BinaryDistribution(Distribution):

    def has_ext_modules(foo):
        return True

# Setup the list of external dlls.
import os
if os.name == 'nt':
    mingw_wheel_libs = 'mingw_wheel_libs_python{}{}.txt'.format(
        sys.version_info[0], sys.version_info[1])
    l = open(mingw_wheel_libs, 'r').readlines()
    DLL_LIST = [os.path.basename(_[:-1]) for _ in l]

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      url=URL,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      classifiers=CLASSIFIERS,
      keywords=KEYWORDS,
      platforms=PLATFORMS,
      install_requires=INSTALL_REQUIRES,
      packages=['pygmo', 'pygmo.plotting'],
      # Include pre-compiled extension
      package_data={'pygmo': ['core.pyd'] + \
                    DLL_LIST if os.name == 'nt' else ['core.so']},
      distclass=BinaryDistribution)
