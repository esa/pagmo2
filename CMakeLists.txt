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

# Build option: enable NLopt.
option(PAGMO_WITH_NLOPT "Enable wrappers for the NLopt algorithms." OFF)

# Build option: install header.
option(PAGMO_INSTALL_HEADERS "Enable the installation of PaGMO's header files." ON)

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
if(YACMA_COMPILER_IS_MSVC)
  # Disable the idiotic minmax macros on MSVC, some annoying warnings,
  # and enable the bigobj option.
  list(APPEND PAGMO_CXX_FLAGS_DEBUG "-DNOMINMAX" "/wd4459" "/wd4127" "/bigobj")
  list(APPEND PAGMO_CXX_FLAGS_RELEASE "-DNOMINMAX" "/wd4459" "/wd4127" "/bigobj")
endif()
if(YACMA_COMPILER_IS_INTELXX)
  # NOTE: on MSVC we use the push/pop pragmas, but they do not seem to work on Intel (the pragmas
  # in icc influence the behaviour at instantiation point, not at definition point).
  # These warnings are useful in principle, but they are generated a lot from cereal and we have no
  # way of disabling them selectively. Just rely on the other compilers to provde good diagnostic.
  list(APPEND PAGMO_CXX_FLAGS_DEBUG "-diag-disable" "2259,1682,68")
  list(APPEND PAGMO_CXX_FLAGS_RELEASE "-diag-disable" "2259,1682,68")
endif()

# Setting up Eigen3
if(PAGMO_WITH_EIGEN3)
    find_package(Eigen3 REQUIRED)
    message(STATUS "Eigen include directory: ${EIGEN3_INCLUDE_DIR}")
    message(STATUS "Eigen version detected: ${EIGEN3_VERSION}")
endif()

# NLopt setup
if(PAGMO_WITH_NLOPT)
    find_package(NLOPT REQUIRED)
    message(STATUS "NLopt include directory: ${NLOPT_INCLUDE_DIRS}")
    message(STATUS "NLopt libraries: ${NLOPT_LIBRARIES}")
endif()

# Python setup.
# NOTE: we do it here because we need to detect the Python bits *before*
# looking for boost.python.
if(PAGMO_BUILD_PYGMO)
    include(YACMAPythonSetup)

    # Python version check.
    if(${PYTHON_VERSION_MAJOR} LESS 2 OR (${PYTHON_VERSION_MAJOR} EQUAL 2 AND ${PYTHON_VERSION_MINOR} LESS 7))
        message(FATAL_ERROR "Minimum supported Python version is 2.7.")
    endif()

    if(NOT YACMA_NUMPY_INCLUDES_DIR)
        message(FATAL_ERROR "The NumPy headers could not be located.")
    endif()
endif()

# Boost setup.
include(PagmoFindBoost)

# Setup of the header-only pagmo library.
add_library(pagmo INTERFACE)
target_link_libraries(pagmo INTERFACE Threads::Threads Boost::boost)
target_include_directories(pagmo INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

if(PAGMO_WITH_EIGEN3)
    # Link pagmo to eigen3.
    add_library(Eigen3::eigen3 INTERFACE IMPORTED)
    set_target_properties(Eigen3::eigen3 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
    target_link_libraries(pagmo INTERFACE Eigen3::eigen3)
    # FIXME: this needs to go into config.hpp, once implemented.
    target_compile_definitions(pagmo INTERFACE PAGMO_WITH_EIGEN3)
endif()

if(PAGMO_WITH_NLOPT)
    # Link pagmo to NLopt.
    add_library(NLOPT::nlopt UNKNOWN IMPORTED)
    set_target_properties(NLOPT::nlopt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NLOPT_INCLUDE_DIRS}")
    set_target_properties(NLOPT::nlopt PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C" IMPORTED_LOCATION "${NLOPT_LIBRARIES}")
    target_link_libraries(pagmo INTERFACE NLOPT::nlopt)
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

if(PAGMO_INSTALL_HEADERS)
  install(DIRECTORY include/ DESTINATION include)
endif()
