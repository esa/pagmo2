#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

if [[ "${PAGMO_BUILD}" != manylinux* ]]; then
    if [[ "${TRAVIS_OS_NAME}" == "osx" ]]; then
        wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O miniconda.sh;
    else
        wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
    fi
    export deps_dir=$HOME/local
    export PATH="$HOME/miniconda/bin:$PATH"
    bash miniconda.sh -b -p $HOME/miniconda
    conda config --add channels conda-forge --force

    conda_pkgs="cmake eigen nlopt ipopt boost boost-cpp tbb tbb-devel"

    if [[ "${PAGMO_BUILD}" == "Python37" || "${PAGMO_BUILD}" == "OSXPython37" ]]; then
        conda_pkgs="$conda_pkgs python=3.7 numpy cloudpickle dill ipyparallel numba pip"
    elif [[ "${PAGMO_BUILD}" == "Python27" || "${PAGMO_BUILD}" == "OSXPython27" ]]; then
        conda_pkgs="$conda_pkgs python=2.7 numpy cloudpickle dill ipyparallel numba"
    fi

    if [[ "${PAGMO_BUILD}" == Python* ]]; then
        conda_pkgs="$conda_pkgs graphviz doxygen"
    fi

    conda create -q -p $deps_dir -y
    source activate $deps_dir
    # NOTE: install the GCC 4.8 version of the conda packages,
    # otherwise we have errors which I think are related to
    # ABI issues.
    conda install -c conda-forge/label/cf201901 $conda_pkgs -y
fi

set +e
set +x
