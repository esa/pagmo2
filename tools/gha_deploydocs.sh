#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniforge/bin:$PATH"
bash miniforge.sh -b -p $HOME/miniforge
conda create -y -q -p $deps_dir c-compiler cxx-compiler cmake eigen nlopt ipopt boost-cpp tbb tbb-devel python=3.10 sphinx=4.5.0 sphinx-book-theme breathe "doxygen<1.13" graphviz
source activate $deps_dir

## Create the build dir and cd into it.
cd ${GITHUB_WORKSPACE} # not necessary?
mkdir build
cd build

# GCC build.
cmake ../ -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Release -DPAGMO_BUILD_TESTS=yes -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes -DPAGMO_ENABLE_IPO=yes

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

set +e
set +x
