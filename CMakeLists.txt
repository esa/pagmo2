project(pagmo)

enable_testing()

cmake_minimum_required(VERSION 3.2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules" "${CMAKE_SOURCE_DIR}/cmake_modules/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")

# Set default build type to "Release".
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

# Build option: enable test set.
option(PAGMO_BUILD_TESTS "Build test set." OFF)

# Build option: enable tutorials.
option(PAGMO_BUILD_TUTORIALS "Build tutorials." OFF)

# Build option: enable PyGMO.
option(PAGMO_BUILD_PYGMO "Build PyGMO." OFF)

# Build option: enable features depending on Eigen3.
option(PAGMO_WITH_EIGEN3 "Enable features depending on Eigen3 (such as CMAES). Requires Eigen3." OFF)

# Initial setup of compiler flags.
include(YACMACompilerLinkerSettings)

# Threading setup.
include(YACMAThreadingSetup)

# Assemble the flags.
set(PAGMO_CXX_FLAGS_DEBUG ${YACMA_CXX_FLAGS} ${YACMA_CXX_FLAGS_DEBUG} ${YACMA_THREADING_CXX_FLAGS})
set(PAGMO_CXX_FLAGS_RELEASE ${YACMA_CXX_FLAGS} ${YACMA_THREADING_CXX_FLAGS})
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND YACMA_COMPILER_IS_CLANGXX)
  message(STATUS "Clang compiler on OSX detected, setting the standard library to 'libc++'.")
  list(APPEND PAGMO_CXX_FLAGS_DEBUG "-stdlib=libc++")
  list(APPEND PAGMO_CXX_FLAGS_RELEASE "-stdlib=libc++")
endif()

# Setting up Eigen3
if(PAGMO_WITH_EIGEN3)
    find_package(Eigen3 REQUIRED)
    message(STATUS "Eigen include directory: ${EIGEN3_INCLUDE_DIR}")
    message(STATUS "Eigen version detected: ${EIGEN3_VERSION}")
endif()

# Setup of the header-only pagmo library.
add_library(pagmo INTERFACE)
target_link_libraries(pagmo INTERFACE Threads::Threads)
target_include_directories(pagmo INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
if(PAGMO_WITH_EIGEN3)
    target_include_directories(pagmo SYSTEM INTERFACE "${EIGEN3_INCLUDE_DIR}")
    target_compile_definitions(pagmo INTERFACE PAGMO_WITH_EIGEN3)
endif()

if(PAGMO_BUILD_TESTS)
    add_subdirectory("${CMAKE_SOURCE_DIR}/tests")
endif()

if(PAGMO_BUILD_TUTORIALS)
    add_subdirectory("${CMAKE_SOURCE_DIR}/tutorials")
endif()

if(PAGMO_BUILD_PYGMO)
    add_subdirectory("${CMAKE_SOURCE_DIR}/pygmo")
endif()
