#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install build-essential wget

# Install conda+deps.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
export PATH="$deps_dir/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge --force
conda_pkgs="cmake eigen nlopt ipopt boost boost-cpp tbb tbb-devel python=3.8 numpy cloudpickle dill numba pip"
conda create -q -p $deps_dir -y
source activate $deps_dir
conda install $conda_pkgs -y

# Build/install pagmo.
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DPAGMO_BUILD_TESTS=no -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes -DBoost_NO_BOOST_CMAKE=ON
make -j2 VERBOSE=1 install

# Build/install pygmo.
cd ..
mkdir build_py
cd build_py
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DPAGMO_BUILD_PAGMO=no -DPAGMO_BUILD_PYGMO=yes -DBoost_NO_BOOST_CMAKE=ON
make -j2 VERBOSE=1 install

# Run the tests.
cd ../tools
python -c "import pygmo; pygmo.test.run_test_suite(1); pygmo.mp_island.shutdown_pool(); pygmo.mp_bfe.shutdown_pool()";

# Additional serialization tests.
python travis_additional_tests.py;

# AP examples.
cd ../ap_examples/uda_basic;
mkdir build;
cd build;
cmake -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DBoost_NO_BOOST_CMAKE=ON ../;
make install VERBOSE=1;
cd ../../;
python test1.py

cd udp_basic;
mkdir build;
cd build;
cmake -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DBoost_NO_BOOST_CMAKE=ON ../;
make install VERBOSE=1;
cd ../../;
python test2.py

set +e
set +x
