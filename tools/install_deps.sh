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

conda_pkgs="boost-cpp>=1.55 cmake>=3.2 eigen"

if [[ "${PAGMO_BUILD}" == "PygmoPython35" ]]; then
    conda_pkgs="$conda_pkgs boost>=1.55 python=3.5 numpy dill ipyparallel"
fi

conda create -q -p $deps_dir -y $conda_pkgs
source activate $deps_dir

set +e
set +x
