#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install build-essential cmake libboost-dev libboost-serialization-dev libboost-test-dev libnlopt-dev libeigen3-dev coinor-libipopt-dev curl libtbb-dev

# Create the build dir and cd into it.
mkdir build
cd build

# GCC build with coverage.
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DPAGMO_BUILD_TESTS=yes -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes -DCMAKE_CXX_FLAGS="--coverage"
make -j2 VERBOSE=1
ctest -V

# Upload coverage data.
bash <(curl -s https://codecov.io/bash) -x gcov-7

set +e
set +x
