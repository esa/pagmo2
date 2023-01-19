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
conda install mamba
mamba create -y -q -p $deps_dir c-compiler cxx-compiler cmake eigen nlopt ipopt boost-cpp tbb tbb-devel python=3.10 sphinx=4.5.0 sphinx-book-theme breathe doxygen graphviz
source activate $deps_dir

## Create the build dir and cd into it.
cd ${GITHUB_WORKSPACE} # not necessary?
mkdir build
cd build

# GCC build.
cmake ../ -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Release -DPAGMO_BUILD_TESTS=yes -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes -DPAGMO_ENABLE_IPO=yes
make -j4 VERBOSE=1
ctest -V -j4

# Build the documentation.
# Doxygen.
cd ../doc/doxygen
export DOXYGEN_OUTPUT=`doxygen 2>&1 >/dev/null`;
if [[ "${DOXYGEN_OUTPUT}" != "" ]]; then
    echo "Doxygen encountered some problem:";
    echo "${DOXYGEN_OUTPUT}";
    exit 1;
fi
echo "Doxygen ran successfully";

# Copy the images into the xml output dir (this is needed by sphinx).
cp images/* xml/;
cd ../sphinx/;
make html linkcheck

# Run the doctests.
make doctest;

set +e
set +x