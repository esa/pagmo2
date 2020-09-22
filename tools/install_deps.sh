#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O miniconda.sh;

export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge
conda config --set channel_priority strict

conda_pkgs="cmake eigen nlopt ipopt boost boost-cpp tbb tbb-devel"

conda create -q -p $deps_dir -y
source activate $deps_dir
conda install $conda_pkgs -y

set +e
set +x
