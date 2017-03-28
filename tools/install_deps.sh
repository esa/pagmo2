#!/usr/bin/env bash

# Exit on error
set -e
# Echo each command
set -x

if [[ "${TRAVIS_OS_NAME}" == "osx" ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O miniconda.sh;
else
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
fi
export deps_dir=$HOME/local
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda config --add channels conda-forge --force

conda_pkgs="boost>=1.55 cmake>=3.2 eigen"

if [[ "${PAGMO_BUILD}" == "Python36" || "${PAGMO_BUILD}" == "OSXPython36" ]]; then
    conda_pkgs="$conda_pkgs python=3.6 numpy dill ipyparallel"
elif [[ "${PAGMO_BUILD}" == "Python27" || "${PAGMO_BUILD}" == "OSXPython27" ]]; then
    conda_pkgs="$conda_pkgs python=2.7 numpy dill ipyparallel"
fi

if [[ "${PAGMO_BUILD}" == "Python36" || "${PAGMO_BUILD}" == "Python27" ]]; then
    conda_pkgs="$conda_pkgs graphviz"
fi

conda create -q -p $deps_dir -y $conda_pkgs
source activate $deps_dir

set +e
set +x
