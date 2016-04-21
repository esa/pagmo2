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

# Compiler setup.
include(PAGMOCompilerLinkerSettings)
message(STATUS "Current CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Current CXX debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")

# Initial thread setup.
find_package(Threads REQUIRED)
message(STATUS "Thread library: ${CMAKE_THREAD_LIBS_INIT}")
set(MANDATORY_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

if(BUILD_TESTS)
    # Boost unit test library.
    find_package(Boost 1.55.0 REQUIRED COMPONENTS "unit_test_framework")
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
