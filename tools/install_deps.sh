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

    conda_pkgs="cmake>=3.2 eigen nlopt ipopt"
    if [[ "${PAGMO_BUILD}" == "Python36" ]]; then
        # NOTE: for a specific build, Python36, we are pinning the boost and boost-cpp packages versions to the current
        # pinned conda-forge versions, which can be found out at:
        # https://github.com/conda-forge/conda-forge.github.io/blob/master/scripts/pin_the_slow_way.py
        # This ensures that we build with the same boost version we use to build the conda packages.
        # NOTE: these version numbers need to be updated manually.
        conda_pkgs="$conda_pkgs boost=1.65.1 boost-cpp=1.65.1"
    else
        # For the other builds, we pick the latest boost available in conda, in order
        # to ensure better coverage of boost versions.
        conda_pkgs="$conda_pkgs boost boost-cpp"
    fi

    if [[ "${PAGMO_BUILD}" == "Python36" || "${PAGMO_BUILD}" == "OSXPython36" ]]; then
        conda_pkgs="$conda_pkgs python=3.6 numpy cloudpickle ipyparallel numba"
    elif [[ "${PAGMO_BUILD}" == "Python27" || "${PAGMO_BUILD}" == "OSXPython27" ]]; then
        conda_pkgs="$conda_pkgs python=2.7 numpy cloudpickle ipyparallel numba"
    fi

    if [[ "${PAGMO_BUILD}" == Python* ]]; then
        conda_pkgs="$conda_pkgs graphviz doxygen"
    fi

    conda create -q -p $deps_dir -y $conda_pkgs
    source activate $deps_dir
fi

set +e
set +x
