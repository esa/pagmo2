#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Install conda+deps.
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -y -q -p $deps_dir c-compiler cxx-compiler libcxx cmake eigen nlopt ipopt boost-cpp tbb tbb-devel
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# GCC build.
CXX=clang++ CC=clang cmake ../ -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DBoost_NO_BOOST_CMAKE=ON -DPAGMO_BUILD_TESTS=yes -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes -DPAGMO_ENABLE_IPO=yes
make -j2 VERBOSE=1
ctest -V -j4

set +e
set +x
