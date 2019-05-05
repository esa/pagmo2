#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install build-essential cmake libboost-dev libboost-serialization-dev libboost-test-dev libnlopt-dev libeigen3-dev coinor-libipopt-dev clang libtbb-dev

# Create the build dir and cd into it.
mkdir build
cd build

# Clang release build.
CXX=clang++ CC=clang cmake ../ -DCMAKE_CXX_STANDARD=14 -DCMAKE_BUILD_TYPE=Release -DPAGMO_BUILD_TESTS=yes -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes
make -j2 VERBOSE=1
ctest -V

set +e
set +x
