#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install build-essential cmake libboost-dev libnlopt-dev libeigen3-dev coinor-libipopt-dev

# Create the build dir and cd into it.
mkdir build
cd build

# GCC build with address sanitizer.
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DPAGMO_BUILD_TESTS=yes -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes -DCMAKE_CXX_FLAGS="-fsanitize=address"
make -j2 VERBOSE=1
# Run the tests, except the fork island which
# gives spurious warnings in the address sanitizer.
ctest -E fork_island

set +e
set +x
