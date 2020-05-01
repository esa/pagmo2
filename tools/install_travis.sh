#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
export PATH="$deps_dir/bin:$PATH"

if [[ "${PAGMO_BUILD}" == "OSXDebug" ]]; then
    CXX=clang++ CC=clang cmake -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DPAGMO_BUILD_TESTS=yes -DPAGMO_BUILD_TUTORIALS=yes -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes -DCMAKE_CXX_FLAGS_DEBUG="-g0 -Os" ../;
    make -j2 VERBOSE=1;
    ctest -VV;

    cd ..
    mkdir build_static
    cd build_static

    CXX=clang++ CC=clang cmake -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DPAGMO_BUILD_TESTS=yes -DPAGMO_BUILD_TUTORIALS=yes -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes -DPAGMO_BUILD_STATIC_LIBRARY=yes -DCMAKE_CXX_FLAGS_DEBUG="-g0 -Os" ../;
    make -j2 VERBOSE=1;
    ctest -VV;
elif [[ "${PAGMO_BUILD}" == "OSXRelease" ]]; then
    CXX=clang++ CC=clang cmake -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Release -DPAGMO_BUILD_TESTS=yes -DPAGMO_BUILD_TUTORIALS=yes -DPAGMO_WITH_EIGEN3=yes -DPAGMO_WITH_NLOPT=yes -DPAGMO_WITH_IPOPT=yes -DPAGMO_ENABLE_IPO=yes ../;
    make -j2 VERBOSE=1;
    ctest -VV;
fi

set +e
set +x
