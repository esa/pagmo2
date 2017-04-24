cmake_minimum_required(VERSION 3.2)

project(pagmo VERSION 2.2)

enable_testing()

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

# Build option: enable Ipopt.
option(PAGMO_WITH_IPOPT "Enable wrappers for the Ipopt solver." OFF)

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
if(MINGW)
	# Flag needed to deal with big binaries in MinGW.
	message(STATUS "Enabling the '-Wa,-mbig-obj' flag in MinGW builds.")
	list(APPEND PAGMO_CXX_FLAGS_DEBUG "-Wa,-mbig-obj")
	list(APPEND PAGMO_CXX_FLAGS_RELEASE "-Wa,-mbig-obj")
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
    message(STATUS "NLopt include directory: ${NLOPT_INCLUDE_DIR}")
    message(STATUS "NLopt library: ${NLOPT_LIBRARY}")
endif()

# Ipopt setup
if(PAGMO_WITH_IPOPT)
    find_package(IPOPT REQUIRED)
    message(STATUS "Ipopt include directory: ${IPOPT_INCLUDE_DIRS}")
    message(STATUS "Ipopt libraries: ${IPOPT_LIBRARIES}")
    message(STATUS "Ipopt definitions: ${IPOPT_DEFINITIONS}")
    message(STATUS "Ipopt link flags: ${IPOPT_LINK_FLAGS}")
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

    if(NOT YACMA_NUMPY_INCLUDE_DIR)
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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

if(PAGMO_WITH_EIGEN3)
    # Link pagmo to eigen3.
    add_library(Eigen3::eigen3 INTERFACE IMPORTED)
    set_target_properties(Eigen3::eigen3 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
    target_link_libraries(pagmo INTERFACE Eigen3::eigen3)
    set(PAGMO_ENABLE_EIGEN3 "#define PAGMO_WITH_EIGEN3")
endif()

if(PAGMO_WITH_NLOPT)
    # Link pagmo to NLopt.
    target_link_libraries(pagmo INTERFACE NLOPT::nlopt)
    set(PAGMO_ENABLE_NLOPT "#define PAGMO_WITH_NLOPT")
endif()

if(PAGMO_WITH_IPOPT)
    # Link pagmo to Ipopt.
    add_library(IPOPT::ipopt INTERFACE IMPORTED)
    set_target_properties(IPOPT::ipopt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${IPOPT_INCLUDE_DIRS}")
    # NOTE: I have no idea why this happens, but in addition to the -pthread flag (which should be enough
    # to signal we need pthreads - and indeed it is in any other situation I have encountered so far),
    # libipopt.so apprently really really strongly wants '-lpthread' to be linked in explicitly at the
    # very beginning. Why oh why scientific software is always so much of a pain to deal with?
    if(CMAKE_USE_PTHREADS_INIT)
        message(STATUS "Explicitly linking ipopt with '-lpthread'.")
        set_property(TARGET IPOPT::ipopt APPEND PROPERTY INTERFACE_LINK_LIBRARIES "pthread")
    endif()
    set_property(TARGET IPOPT::ipopt APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${IPOPT_LIBRARIES}")
    set_target_properties(IPOPT::ipopt PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${IPOPT_DEFINITIONS}")
    # NOTE: this is a tentative workaround to setup custom link flags, if existing. See:
    # https://gitlab.kitware.com/cmake/cmake/issues/16543
    set_property(TARGET IPOPT::ipopt APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${IPOPT_LINK_FLAGS}")
    target_link_libraries(pagmo INTERFACE IPOPT::ipopt)
    set(PAGMO_ENABLE_IPOPT "#define PAGMO_WITH_IPOPT")
endif()

# Configure config.hpp.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/include/pagmo/config.hpp" @ONLY)

# Configure the doc files.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/Doxyfile.in" "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/Doxyfile" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx/conf.py.in" "${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx/conf.py" @ONLY)

if(PAGMO_BUILD_TESTS)
    add_subdirectory("${CMAKE_SOURCE_DIR}/tests")
endif()

if(PAGMO_BUILD_TUTORIALS)
    add_subdirectory("${CMAKE_SOURCE_DIR}/tutorials")
endif()

if(PAGMO_BUILD_PYGMO)
    add_subdirectory("${CMAKE_SOURCE_DIR}/pygmo")
    if(MINGW OR ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        message(STATUS "Creating the files for the generation of a binary wheel.")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tools/wheel_setup.py" "${CMAKE_CURRENT_BINARY_DIR}/wheel/setup.py" @ONLY)
        if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
            # NOTE: this is necessary on linux but harmful on mingw.
            configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tools/wheel_setup.cfg" "${CMAKE_CURRENT_BINARY_DIR}/wheel/setup.cfg" @ONLY)
        endif()
        if(MINGW)
            configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tools/mingw_wheel_libs_python${PYTHON_VERSION_MAJOR}.txt" "${CMAKE_CURRENT_BINARY_DIR}/wheel/mingw_wheel_libs_python${PYTHON_VERSION_MAJOR}.txt" @ONLY)
        endif()
	endif()
endif()

if(PAGMO_INSTALL_HEADERS)
  install(DIRECTORY include/ DESTINATION include)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/pagmo/config.hpp" DESTINATION include/pagmo)
endif()
