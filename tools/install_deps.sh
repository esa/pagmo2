#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

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

conda create -q -p $deps_dir -y
source activate $deps_dir
# NOTE: install the GCC 4.8 version of the conda packages,
# otherwise we have errors which I think are related to
# ABI issues.
conda install -c conda-forge/label/cf201901 $conda_pkgs -y

set +e
set +x
