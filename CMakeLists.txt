# CMake version check.
cmake_minimum_required(VERSION 3.8)
# NOTE: policy for using the CheckIPOSupported module:
# https://cmake.org/cmake/help/latest/policy/CMP0069.html
cmake_policy(SET CMP0069 NEW)

# Set default build type to "Release".
# NOTE: this should be done before the project command since the latter can set
# CMAKE_BUILD_TYPE itself (it does so for nmake).
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

project(pagmo VERSION 2.19.0 LANGUAGES CXX C)

# Setup module path.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
enable_testing()

# Build option: enable the test suite.
option(PAGMO_BUILD_TESTS "Build the test suite." OFF)

# Build option: enable the benchmarks.
option(PAGMO_BUILD_BENCHMARKS "Build the benchmarks." OFF)

# Build option: enable tutorials.
option(PAGMO_BUILD_TUTORIALS "Build tutorials." OFF)

# Build option: enable features depending on Eigen3.
option(PAGMO_WITH_EIGEN3 "Enable features depending on Eigen3 (such as CMAES). Requires Eigen3." OFF)

# Build option: enable NLopt.
option(PAGMO_WITH_NLOPT "Enable wrappers for the NLopt algorithms." OFF)

# Build option: enable Ipopt.
option(PAGMO_WITH_IPOPT "Enable wrappers for the Ipopt solver." OFF)

# Build option: enable IPO.
option(PAGMO_ENABLE_IPO "Enable IPO (requires CMake >= 3.9 and compiler support)." OFF)
mark_as_advanced(PAGMO_ENABLE_IPO)

# Build static library instead of dynamic.
option(PAGMO_BUILD_STATIC_LIBRARY "Build pagmo as a static library, instead of dynamic." OFF)

# Detect if we can enable the fork_island UDI.
include(CheckIncludeFileCXX)
include(CheckCXXSymbolExists)
CHECK_INCLUDE_FILE_CXX("sys/types.h" PAGMO_HAVE_SYS_TYPES_H)
CHECK_INCLUDE_FILE_CXX("sys/wait.h" PAGMO_HAVE_SYS_WAIT_H)
CHECK_INCLUDE_FILE_CXX("unistd.h" PAGMO_HAVE_UNISTD_H)
CHECK_CXX_SYMBOL_EXISTS(fork "unistd.h;sys/types.h" PAGMO_HAVE_FORK_SYSCALL)
if(PAGMO_HAVE_SYS_TYPES_H AND PAGMO_HAVE_SYS_WAIT_H AND PAGMO_HAVE_UNISTD_H AND PAGMO_HAVE_FORK_SYSCALL)
    message(STATUS "The fork_island UDI will be available.")
    set(PAGMO_WITH_FORK_ISLAND YES)
    set(PAGMO_ENABLE_FORK_ISLAND "#define PAGMO_WITH_FORK_ISLAND")
else()
    message(STATUS "The fork_island UDI will NOT be available.")
    set(PAGMO_WITH_FORK_ISLAND NO)
endif()

# Threading setup.
# NOTE: do it before checking for the presence of pthread_atfork(),
# since we need to know if we are using pthreads.
include(YACMAThreadingSetup)

# If the fork() system call is supported and we are using pthreads,
# check if the pthread_atfork() function is available. This is used
# to clean up problematic global variables (such as the task queue
# cache) when a process is forked.
if(PAGMO_HAVE_FORK_SYSCALL AND CMAKE_USE_PTHREADS_INIT)
    set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
    CHECK_CXX_SYMBOL_EXISTS(pthread_atfork "pthread.h" PAGMO_HAVE_PTHREAD_ATFORK)
    unset(CMAKE_REQUIRED_LIBRARIES)
endif()

# Initial setup of compiler flags.
include(YACMACompilerLinkerSettings)
include(CheckCXXCompilerFlag)

# NOTE: on Unix systems, the correct library installation path
# could be something other than just "lib", such as "lib64",
# "lib32", etc., depending on platform/configuration. Apparently,
# CMake provides this information via the GNUInstallDirs module.
# Let's enable this for now on all Unixes except OSX.
# NOTE: potentially, this could be applicable to Cygwin as well.
#
# https://cmake.org/cmake/help/v3.15/module/GNUInstallDirs.html
# https://cmake.org/pipermail/cmake/2013-July/055375.html
if(UNIX AND NOT APPLE)
    include(GNUInstallDirs)
    set(_PAGMO_INSTALL_LIBDIR_DEFAULT "${CMAKE_INSTALL_LIBDIR}")
else()
    set(_PAGMO_INSTALL_LIBDIR_DEFAULT "lib")
endif()
if(NOT PAGMO_INSTALL_LIBDIR)
    set(PAGMO_INSTALL_LIBDIR "${_PAGMO_INSTALL_LIBDIR_DEFAULT}" CACHE STRING
        "Library installation directory." FORCE)
endif()
mark_as_advanced(PAGMO_INSTALL_LIBDIR)
message(STATUS "Library installation directory: ${PAGMO_INSTALL_LIBDIR}")

# Assemble the flags.
set(PAGMO_CXX_FLAGS_DEBUG ${YACMA_CXX_FLAGS} ${YACMA_CXX_FLAGS_DEBUG} ${YACMA_THREADING_CXX_FLAGS})
set(PAGMO_CXX_FLAGS_RELEASE ${YACMA_CXX_FLAGS} ${YACMA_THREADING_CXX_FLAGS})
if(APPLE AND YACMA_COMPILER_IS_CLANGXX)
  message(STATUS "Clang compiler on OSX detected, setting the standard library to 'libc++'.")
  list(APPEND PAGMO_CXX_FLAGS_DEBUG "-stdlib=libc++")
  list(APPEND PAGMO_CXX_FLAGS_RELEASE "-stdlib=libc++")
endif()
if(YACMA_COMPILER_IS_MSVC)
  # Disable the idiotic minmax macros on MSVC, some annoying warnings,
  # enable the bigobj option and the WIN32_LEAN_AND_MEAN definition.
  list(APPEND PAGMO_CXX_FLAGS_DEBUG "-DNOMINMAX" "/wd4459" "/wd4127" "/wd4702" "/wd4251" "/wd4701" "/bigobj" "-DWIN32_LEAN_AND_MEAN")
  list(APPEND PAGMO_CXX_FLAGS_RELEASE "-DNOMINMAX" "/wd4459" "/wd4127" "/wd4702" "/wd4251" "/wd4701" "/bigobj" "-DWIN32_LEAN_AND_MEAN")
  # Enable strict conformance mode, if supported.
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_compiler_flag("/permissive-" _PAGMO_MSVC_SUPPORTS_STRICT_CONFORMANCE)
  unset(CMAKE_REQUIRED_QUIET)
  if(_PAGMO_MSVC_SUPPORTS_STRICT_CONFORMANCE)
    message(STATUS "The '/permissive-' flag is supported, enabling it.")
    list(APPEND PAGMO_CXX_FLAGS_DEBUG "/permissive-")
    list(APPEND PAGMO_CXX_FLAGS_RELEASE "/permissive-")
  endif()
  unset(_PAGMO_MSVC_SUPPORTS_STRICT_CONFORMANCE)
endif()
if(MINGW)
	# Flag needed to deal with big binaries in MinGW.
	message(STATUS "Enabling the '-Wa,-mbig-obj' flag in MinGW builds.")
	list(APPEND PAGMO_CXX_FLAGS_DEBUG "-Wa,-mbig-obj")
	list(APPEND PAGMO_CXX_FLAGS_RELEASE "-Wa,-mbig-obj")
endif()
# NOTE: at least up to version 7, GCC is needlessly chatty
# about the 'override' attribute. Thus, we manually disable
# the -Wsuggest-override debug flag.
if(YACMA_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8")
    include(CheckCXXCompilerFlag)
    set(CMAKE_REQUIRED_QUIET TRUE)
    check_cxx_compiler_flag("-Wno-suggest-override" _PAGMO_GCC_SUPPORTS_NO_OVERRIDE)
    unset(CMAKE_REQUIRED_QUIET)
    if(_PAGMO_GCC_SUPPORTS_NO_OVERRIDE)
        message(STATUS "Enabling the '-Wno-suggest-override' flag for GCC<8.")
        list(APPEND PAGMO_CXX_FLAGS_DEBUG "-Wno-suggest-override")
    endif()
    unset(_PAGMO_GCC_SUPPORTS_NO_OVERRIDE)
endif()

# TBB. Try to find it first in config mode (supported
# since version 2021 after the oneTBB rename), and, if this
# fails, fall back to our own FindTBB.cmake. This is of course
# not an ideal solution, but it should work until oneTBB
# becomes widely deployed.
find_package(TBB QUIET CONFIG)
if(NOT TBB_FOUND)
    message(STATUS "TBB not found using config mode, retrying in module mode.")
    find_package(TBB REQUIRED MODULE)
else()
    message(STATUS "TBB found using config mode.")
endif()

# Eigen3
if(PAGMO_WITH_EIGEN3)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# NLopt
if(PAGMO_WITH_NLOPT)
    find_package(NLopt 2.6 REQUIRED NO_MODULE)
endif()

# Ipopt
if(PAGMO_WITH_IPOPT)
    find_package(pagmo_IPOPT REQUIRED COMPONENTS header libipopt)
endif()

if(PAGMO_BUILD_TESTS)
    # Internal variable that will be used to tell PagmoFindBoost to locate the
    # Boost unit test framework, if tests are required.
    set(_PAGMO_FIND_BOOST_UNIT_TEST_FRAMEWORK TRUE)
endif()

# Boost setup.
include(PagmoFindBoost)

# List of source files.
set(PAGMO_SRC_FILES
    # Core classes.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/population.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problem.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bfe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/island.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/archipelago.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/threading.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/topology.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/r_policy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/s_policy.cpp"
    # UDP.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/null_problem.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2006.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2009.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/schwefel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/rosenbrock.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/hock_schittkowski_71.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/inventory.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/zdt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/dtlz.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/unconstrain.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/translate.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/decompose.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/golomb_ruler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/lennard_jones.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/ackley.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/griewank.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/rastrigin.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/minlp_rastrigin.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/luksan_vlcek1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/wfg.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2013.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2013_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2014.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2014_data.cpp"
    # UDA.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/null_algorithm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/de.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/pso.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/not_population_based.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/compass_search.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/mbh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/cstrs_self_adaptive.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/pso_gen.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/ihs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/sade.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/bee_colony.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/sea.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/sga.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/simulated_annealing.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/moead.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/moead_gen.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/nsga2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/gaco.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/de1220.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/gwo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/maco.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/nspso.cpp"
    # UDI.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/islands/thread_island.cpp"
    # UDBFE.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/batch_evaluators/default_bfe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/batch_evaluators/member_bfe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/batch_evaluators/thread_bfe.cpp"
    # UDT.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/topologies/base_bgl_topology.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/topologies/unconnected.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/topologies/fully_connected.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/topologies/ring.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/topologies/free_form.cpp"
    # UDRP.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/r_policies/fair_replace.cpp"
    # UDSP.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/s_policies/select_best.cpp"
    # Utils.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/constrained.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/discrepancy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/generic.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/genetic_operators.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/multi_objective.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/hypervolume.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/hv_algos/hv_algorithm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/hv_algos/hv_bf_approx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/hv_algos/hv_bf_fpras.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/hv_algos/hv_hv2d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/hv_algos/hv_hv3d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/hv_algos/hv_hvwfg.cpp"
    # Detail.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/base_sr_policy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/bfe_impl.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/task_queue.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/prime_numbers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/gte_getter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/type_name.cpp"
)

# Optional and platform-dependent bits.
if(PAGMO_WITH_FORK_ISLAND)
    set(PAGMO_SRC_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/islands/fork_island.cpp"
        "${PAGMO_SRC_FILES}"
    )
endif()

if(PAGMO_WITH_EIGEN3)
    set(PAGMO_SRC_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/cmaes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/xnes.cpp"
        "${PAGMO_SRC_FILES}"
    )
endif()

if(PAGMO_WITH_NLOPT)
    set(PAGMO_SRC_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/nlopt.cpp"
        "${PAGMO_SRC_FILES}"
    )
endif()

if(PAGMO_WITH_IPOPT)
    set(PAGMO_SRC_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/ipopt.cpp"
        "${PAGMO_SRC_FILES}"
    )
endif()

# Setup of the pagmo library.
if(PAGMO_BUILD_STATIC_LIBRARY)
    # Setup of the pagmo static library.
    message(STATUS "pagmo will be built as a static library.")
    set(PAGMO_STATIC_BUILD "#define PAGMO_STATIC_BUILD")
    add_library(pagmo STATIC "${PAGMO_SRC_FILES}")
else()
    add_library(pagmo SHARED "${PAGMO_SRC_FILES}")
    set_property(TARGET pagmo PROPERTY VERSION "9.0")
    set_property(TARGET pagmo PROPERTY SOVERSION 9)
    set_target_properties(pagmo PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(pagmo PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)
endif()

# Setup common to both static and shared variants.
target_compile_options(pagmo PRIVATE
    "$<$<CONFIG:Debug>:${PAGMO_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${PAGMO_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${PAGMO_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${PAGMO_CXX_FLAGS_RELEASE}>"
)

# Set the minimum C++ standard to C++17, both when building
# and consuming pagmo.
target_compile_features(pagmo PUBLIC cxx_std_17)
# Enforce vanilla C++17 when compiling pagmo.
set_property(TARGET pagmo PROPERTY CXX_EXTENSIONS NO)
# NOTE: make sure the include directories from the current build
# are included first, so that if there is already a pagmo installation
# in the prefix path we don't risk including the headers from that
# one instead.
target_include_directories(pagmo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0")
    if (PAGMO_ENABLE_IPO)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT _PAGMO_IPO_RESULT OUTPUT _PAGMO_IPO_OUTPUT)
        if (_PAGMO_IPO_RESULT)
            message(STATUS "IPO requested and supported, enabling.")
            set_property(TARGET pagmo PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        else()
            message(STATUS "IPO requested, but it is not supported by the compiler:\n${_PAGMO_IPO_OUTPUT}")
        endif()
        unset(_PAGMO_IPO_RESULT)
        unset(_PAGMO_IPO_OUTPUT)
    endif()
endif()

# Threads.
target_link_libraries(pagmo PUBLIC Threads::Threads)

# Boost.
target_link_libraries(pagmo PUBLIC Boost::boost Boost::serialization)
# NOTE: quench warnings from Boost when building the library.
target_compile_definitions(pagmo PRIVATE BOOST_ALLOW_DEPRECATED_HEADERS)

# TBB.
# NOTE: TBB is a private dependency because
# all uses of TBB are limited to the .cpp files,
# thus fully encapsulated in the pagmo library.
target_link_libraries(pagmo PRIVATE TBB::tbb)

if(PAGMO_WITH_EIGEN3)
    # Link pagmo to eigen3.
    target_link_libraries(pagmo PUBLIC Eigen3::Eigen)
    set(PAGMO_ENABLE_EIGEN3 "#define PAGMO_WITH_EIGEN3")
endif()

if(PAGMO_WITH_NLOPT)
    # Link pagmo to NLopt.
    # NOTE: we make use of some NLopt types in the
    # public pagmo interface.
    target_link_libraries(pagmo PUBLIC NLopt::nlopt)
    set(PAGMO_ENABLE_NLOPT "#define PAGMO_WITH_NLOPT")
endif()

if(PAGMO_WITH_IPOPT)
    # Link pagmo to Ipopt.
    # NOTE: we make use of some Ipopt types in the
    # public pagmo interface, but we don't use any
    # function from the library in the public headers.
    # Thus, depend publicly on the header,
    # privately on the compiled library.
    target_link_libraries(pagmo PUBLIC pagmo::IPOPT::header)
    target_link_libraries(pagmo PRIVATE pagmo::IPOPT::libipopt)
    set(PAGMO_ENABLE_IPOPT "#define PAGMO_WITH_IPOPT")
endif()

# Configure config.hpp.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/include/pagmo/config.hpp" @ONLY)

# Configure the doc files.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/Doxyfile.in" "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/Doxyfile" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx/conf.py.in" "${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx/conf.py" @ONLY)

# This is just a simple counter variable, internal use only.
set(_PAGMO_TEST_NUM "0")
# Check splitting options. These need to be set from the command line.
# - PAGMO_TEST_NSPLIT: number of chunks into which the unit tests will be divided (must be > 1).
# - PAGMO_TEST_SPLIT_NUM: 0-based index of the chunk to run.
if(PAGMO_TEST_NSPLIT AND "${PAGMO_TEST_SPLIT_NUM}" STREQUAL "")
    message(FATAL_ERROR "Test splitting was requested, but the PAGMO_TEST_SPLIT_NUM variable was not set.")
elseif(NOT PAGMO_TEST_NSPLIT AND NOT "${PAGMO_TEST_SPLIT_NUM}" STREQUAL "")
    message(FATAL_ERROR "The PAGMO_TEST_SPLIT_NUM variable was set, but test splitting was not requested.")
endif()
if(PAGMO_TEST_NSPLIT)
    message(STATUS "Tests will be split into ${PAGMO_TEST_NSPLIT} chunks. The chunk with index ${PAGMO_TEST_SPLIT_NUM} will be processed.")
endif()

if(PAGMO_BUILD_TESTS)
    add_subdirectory("${CMAKE_SOURCE_DIR}/tests")
endif()

if(PAGMO_BUILD_BENCHMARKS)
    add_subdirectory("${CMAKE_SOURCE_DIR}/benchmarks")
endif()

if(PAGMO_BUILD_TUTORIALS)
    add_subdirectory("${CMAKE_SOURCE_DIR}/tutorials")
endif()

# Library installation.
# Installation of the header files.
install(DIRECTORY include/ DESTINATION include)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/pagmo/config.hpp" DESTINATION include/pagmo)

# Installation of the library.
install(TARGETS pagmo
    EXPORT pagmo_export
    LIBRARY DESTINATION "${PAGMO_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${PAGMO_INSTALL_LIBDIR}"
    RUNTIME DESTINATION bin
)

# Setup of the optional deps.
set(_PAGMO_CONFIG_OPTIONAL_DEPS)
if(PAGMO_WITH_EIGEN3)
    set(_PAGMO_CONFIG_OPTIONAL_DEPS "${_PAGMO_CONFIG_OPTIONAL_DEPS}find_package(Eigen3 3.3 REQUIRED NO_MODULE)\n")
endif()
if(PAGMO_WITH_NLOPT)
    set(_PAGMO_CONFIG_OPTIONAL_DEPS "${_PAGMO_CONFIG_OPTIONAL_DEPS}find_package(NLopt 2.6 REQUIRED NO_MODULE)\n")
endif()
if(PAGMO_WITH_IPOPT)
    set(_PAGMO_CONFIG_OPTIONAL_DEPS "${_PAGMO_CONFIG_OPTIONAL_DEPS}find_package(pagmo_IPOPT REQUIRED COMPONENTS header)\n")
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pagmo-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/pagmo-config.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pagmo-config.cmake" DESTINATION "${PAGMO_INSTALL_LIBDIR}/cmake/pagmo")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/Findpagmo_IPOPT.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/PagmoFindBoost.cmake"
    DESTINATION "${PAGMO_INSTALL_LIBDIR}/cmake/pagmo")
install(EXPORT pagmo_export NAMESPACE Pagmo:: DESTINATION "${PAGMO_INSTALL_LIBDIR}/cmake/pagmo")
# Take care of versioning.
include(CMakePackageConfigHelpers)
# NOTE: SameMinorVersion available only
# since CMake 3.11.
if(${CMAKE_VERSION} VERSION_LESS "3.11.0")
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/pagmo-config-version.cmake" VERSION ${pagmo_VERSION}
        COMPATIBILITY SameMajorVersion)
else()
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/pagmo-config-version.cmake" VERSION ${pagmo_VERSION}
        COMPATIBILITY SameMinorVersion)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pagmo-config-version.cmake" DESTINATION "${PAGMO_INSTALL_LIBDIR}/cmake/pagmo")

# Uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
