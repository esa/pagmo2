#!/usr/bin/env bash

# Echo each command.
set -x

# Exit on error.
set -e

DEPS_DIR="$HOME/local"

# Install deps in a dedicated conda env.
conda create -y -q -p "$DEPS_DIR" \
    c-compiler cxx-compiler cmake ninja boost-cpp tbb tbb-devel eigen nlopt ipopt lcov \
    python=3.10 sphinx=4.5.0 sphinx-book-theme breathe "doxygen<1.13" graphviz

# Configure, build, and test.
conda run -p "$DEPS_DIR" cmake -S . -B build -G Ninja \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBoost_NO_BOOST_CMAKE=ON \
    -DPAGMO_BUILD_TESTS=ON \
    -DPAGMO_WITH_EIGEN3=ON \
    -DPAGMO_WITH_NLOPT=ON \
    -DPAGMO_WITH_IPOPT=ON \
    -DPAGMO_ENABLE_IPO=OFF \
    -DCMAKE_CXX_FLAGS="--coverage" \
    -DCMAKE_CXX_FLAGS_DEBUG="-g -Og"
conda run -p "$DEPS_DIR" cmake --build build --parallel
conda run -p "$DEPS_DIR" ctest --test-dir build -VV --output-on-failure -j4 -E torture

# Generate lcov report.
(
    cd build
    conda run -p "$DEPS_DIR" lcov --capture --directory . --output-file coverage.info
)
test -s build/coverage.info

# Build doxygen XML consumed by sphinx/breathe.
(
    cd doc/doxygen
    conda run -p "$DEPS_DIR" doxygen
    cp images/* xml/
)

# Build docs and soft-fail on external link flakiness.
(
    cd doc/sphinx
    conda run -p "$DEPS_DIR" make html
    if ! conda run -p "$DEPS_DIR" make linkcheck; then
        echo "Warning: Sphinx linkcheck failed (likely transient external outage); continuing."
    fi
)

set +e
set +x
