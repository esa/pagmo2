# Powershell script
# Install conda environment
conda config --set always_yes yes
conda create --name pagmo cmake eigen nlopt ipopt boost-cpp tbb tbb-devel
conda activate pagmo

mkdir build
cd build

cmake `
    -G "Visual Studio 15 2017 Win64" `
    -DCMAKE_PREFIX_PATH=C:\Miniconda\envs\pagmo `
    -DCMAKE_INSTALL_PREFIX=C:\Miniconda\envs\pagmo `
    -DBoost_NO_BOOST_CMAKE=ON `
    -DPAGMO_WITH_EIGEN3=ON `
    -DPAGMO_WITH_IPOPT=ON `
    -DPAGMO_WITH_NLOPT=ON `
    -DPAGMO_ENABLE_IPO=ON `
    -DPAGMO_BUILD_TESTS=YES `
    -DPAGMO_BUILD_TUTORIALS=YES `
    ..

cmake --build . --config Release --target install
ctest -VV --output-on-failure
