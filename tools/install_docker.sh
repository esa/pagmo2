CMAKE_VERSION="3.8.0"
EIGEN3_VERSION="3.3.3"
BOOST_VERSION="1.63.0"
NLOPT_VERSION="2.4.2"

PYTHON_DIR="cp36-cp36m"

# HACK: for python 3.x, the include directory
# is called 'python3.xm' rather than just 'python3.x'.
# This confuses the build system of Boost.Python, thus
# we create a symlink to 'python3.x'.
cd /opt/python/${PYTHON_DIR}/include
PY_INCLUDE_DIR_NAME=`ls`
# If the include dir ends with 'm', create a symlink
# without the 'm'.
if [[ $PY_INCLUDE_DIR_NAME == *m ]]; then
	ln -s $PY_INCLUDE_DIR_NAME `echo $PY_INCLUDE_DIR_NAME|sed 's/.$//'`
fi

cd
mkdir install
cd install

# CMake
wget https://github.com/Kitware/CMake/archive/v${CMAKE_VERSION}.tar.gz
tar xzf v${CMAKE_VERSION}
cd CMake-${CMAKE_VERSION}/
./configure
gmake -j2
gmake install
cd ..

# Eigen
wget https://github.com/RLovelett/eigen/archive/${EIGEN3_VERSION}.tar.gz
tar xzf ${EIGEN3_VERSION}
cd eigen-${EIGEN3_VERSION}
mkdir build
cd build
cmake ../
make install
cd ..
cd ..

# Boost
wget https://downloads.sourceforge.net/project/boost/boost/${BOOST_VERSION}/boost_`echo ${BOOST_VERSION}|tr "." "_"`.tar.bz2
tar xjf boost_`echo ${BOOST_VERSION}|tr "." "_"`.tar.bz2
cd boost_`echo ${BOOST_VERSION}|tr "." "_"`
sh bootstrap.sh --with-python=/opt/python/${PYTHON_DIR}/bin/python
./bjam --toolset=gcc link=shared threading=multi cxxflags="-std=c++11" variant=release --with-python -j2 install
cd ..

# NLopt
wget http://ab-initio.mit.edu/nlopt/nlopt-${NLOPT_VERSION}.tar.gz
tar xzf nlopt-${NLOPT_VERSION}.tar.gz
cd nlopt-${NLOPT_VERSION}
./configure --enable-shared --disable-static
make -j2
make install
cd ..

