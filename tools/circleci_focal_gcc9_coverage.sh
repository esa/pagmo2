#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -y -q -p $deps_dir c-compiler cxx-compiler cmake eigen nlopt ipopt boost-cpp tbb tbb-devel
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# GCC build with coverage.
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DPAGMO_BUILD_TESTS=yes -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes -DCMAKE_CXX_FLAGS="--coverage" -DPAGMO_ENABLE_IPO=yes
make -j2 VERBOSE=1
# NOTE: skip the torture tests because they take
# a lot of time when code coverage is turned on.
ctest -V -j2 -E torture

# Upload coverage data.
bash <(curl -s https://codecov.io/bash) -x gcov-9

set +e
set +x
