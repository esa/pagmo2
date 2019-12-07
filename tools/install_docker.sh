#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

if [[ ${PAGMO_BUILD} == *37 ]]; then
	PYTHON_DIR="cp37-cp37m"
	BOOST_PYTHON_LIBRARY_NAME="libboost_python37.so"
	PYTHON_VERSION="37"
elif [[ ${PAGMO_BUILD} == *36 ]]; then
	PYTHON_DIR="cp36-cp36m"
	BOOST_PYTHON_LIBRARY_NAME="libboost_python36.so"
	PYTHON_VERSION="36"
elif [[ ${PAGMO_BUILD} == *27mu ]]; then
	PYTHON_DIR="cp27-cp27mu"
	BOOST_PYTHON_LIBRARY_NAME="libboost_python27mu.so"
	PYTHON_VERSION="27"
elif [[ ${PAGMO_BUILD} == *27 ]]; then
	PYTHON_DIR="cp27-cp27m"
	BOOST_PYTHON_LIBRARY_NAME="libboost_python27.so"
	PYTHON_VERSION="27"
else
	echo "Invalid build type: ${PAGMO_BUILD}"
	exit 1
fi

cd
cd install

# Python mandatory deps.
/opt/python/${PYTHON_DIR}/bin/pip install cloudpickle numpy
# Python optional deps.
if [[ ${PAGMO_BUILD} != *27m ]]; then
	# NOTE: do not install the optional deps for the py27m build: some of the deps
	# don't have binary wheels available for py27m, which makes pip try to
	# install them from source (which fails).
	/opt/python/${PYTHON_DIR}/bin/pip install dill ipyparallel
	/opt/python/${PYTHON_DIR}/bin/ipcluster start --daemonize=True
	sleep 20
fi

# pagmo & pygmo
cd /pagmo2
mkdir build_pagmo
cd build_pagmo
cmake -DBoost_NO_BOOST_CMAKE=ON \
	-DPAGMO_WITH_EIGEN3=yes \
	-DPAGMO_WITH_NLOPT=yes \
	-DPAGMO_WITH_IPOPT=yes \
	-DCMAKE_BUILD_TYPE=Release ../;
make install
cd ../build
cmake -DBoost_NO_BOOST_CMAKE=ON \
	-DCMAKE_BUILD_TYPE=Release \
	-DPAGMO_BUILD_PYGMO=yes \
	-DPAGMO_BUILD_PAGMO=no \
	-DBoost_PYTHON${PYTHON_VERSION}_LIBRARY_RELEASE=/usr/local/lib/${BOOST_PYTHON_LIBRARY_NAME} \
	-DPYTHON_EXECUTABLE=/opt/python/${PYTHON_DIR}/bin/python ../;
make -j2 install
cd wheel
# Copy the installed pygmo files, wherever they might be in /usr/local,
# into the current dir.
cp -a `find /usr/local/lib -type d -iname 'pygmo'` ./
# Create the wheel and repair it.
/opt/python/${PYTHON_DIR}/bin/python setup.py bdist_wheel
auditwheel repair dist/pygmo* -w ./dist2
# Try to install it and run the tests.
cd /
/opt/python/${PYTHON_DIR}/bin/pip install /pagmo2/build/wheel/dist2/pygmo*
if [[ ${PAGMO_BUILD} == *27m ]]; then
	# NOTE: for the py27m build we don't have ipyparallel installed,
	# which will make some tests fail. Just try to import pygmo in this case.
	/opt/python/${PYTHON_DIR}/bin/python -c "import pygmo"
else
	/opt/python/${PYTHON_DIR}/bin/python -c "import pygmo; pygmo.test.run_test_suite(1)"
fi

# Upload to pypi. This variable will contain something if this is a tagged build (vx.y.z), otherwise it will be empty.
export PAGMO_RELEASE_VERSION=`echo "${TRAVIS_TAG}"|grep -E 'v[0-9]+\.[0-9]+.*'|cut -c 2-`
if [[ "${PAGMO_RELEASE_VERSION}" != "" ]]; then
    echo "Release build detected, uploading to PyPi."
    /opt/python/${PYTHON_DIR}/bin/pip install twine
	/opt/python/${PYTHON_DIR}/bin/twine upload -u ci4esa /pagmo2/build/wheel/dist2/pygmo*
fi
