#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install build-essential wget clang

# Install conda+deps.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge --force
conda_pkgs="cmake eigen nlopt ipopt boost-cpp tbb tbb-devel python=3.7 sphinx sphinx_rtd_theme breathe"
conda create -q -p $deps_dir -y
source activate $deps_dir
conda install $conda_pkgs -y

# Create the build dir and cd into it.
mkdir build
cd build

# Clang release build.
CXX=clang++ CC=clang cmake ../ -DCMAKE_BUILD_TYPE=Release -DPAGMO_BUILD_TESTS=yes -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes
make -j2 VERBOSE=1
ctest -V

set +e
set +x
