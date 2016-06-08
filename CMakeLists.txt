project(pagmo)

enable_testing()

cmake_minimum_required(VERSION 2.8.10)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules" "${CMAKE_SOURCE_DIR}/cmake_modules/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")

# Set default build type to "Release".
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

# Build option: enable test set.
option(BUILD_TESTS "Build test set." OFF)

# Build option: enable tutorials.
option(BUILD_TUTORIALS "Build tutorials." OFF)

# Build option: enable PyGMO.
option(BUILD_PYGMO "Build PyGMO." OFF)

# Build option: I have Eigen in my system!!
option(PAGMO_ENABLE_EIGEN3 "Eigen3 is in the system, use it. This will activate additional includes such as cmaes.h" OFF)

# Compiler setup.
include(PAGMOCompilerLinkerSettings)

# Threading setup.
include(YACMAThreadingSetup)
set(MANDATORY_LIBRARIES ${MANDATORY_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${YACMA_THREADING_CXX_FLAGS}")

message(STATUS "Current CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Current CXX debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")

# Setting up Eigen3
if(PAGMO_ENABLE_EIGEN3)
    find_package(Eigen3)
    if(EIGEN_FOUND)
        message(STATUS "Eigen include directory: ${EIGEN_INCLUDE_DIRS}")
        message(STATUS "Eigen version detected: ${EIGEN_VERSION}")
    else()
        message(FATAL_ERROR "Cannot find Eigen3, the script FindEigen3.cmake failed.")
    endif()
    ADD_DEFINITIONS(-DPAGMO_ENABLE_EIGEN3)
endif()

# Setup the list of the required Boost libraries.
set(REQUIRED_BOOST_LIBS)

if(BUILD_PYGMO)
    # NOTE: we need to run the python setup macros before actually looking
    # for Boost.Python.
    include(YACMAPythonSetup)
    set(REQUIRED_BOOST_LIBS ${REQUIRED_BOOST_LIBS} python)
endif()

# Now actually look for Boost.
if(REQUIRED_BOOST_LIBS)
    find_package(Boost 1.55.0 REQUIRED COMPONENTS "${REQUIRED_BOOST_LIBS}")
endif()

if(BUILD_TESTS)
    add_subdirectory("${CMAKE_SOURCE_DIR}/tests")
endif()

if(BUILD_TUTORIALS)
    add_subdirectory("${CMAKE_SOURCE_DIR}/tutorials")
endif()

if(BUILD_PYGMO)
    add_subdirectory("${CMAKE_SOURCE_DIR}/pygmo")
endif()

# Install the headers.
#install(FILES src/arbpp.hpp DESTINATION include/arbpp)
