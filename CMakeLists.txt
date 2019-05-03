# CMake version check.
cmake_minimum_required(VERSION 3.2)

# Module path setup.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules" "${CMAKE_SOURCE_DIR}/cmake_modules/yacma")

# Set default build type to "Release".
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

# Main build options: build pagmo or pygmo. They cannot be on at the same time,
# and only one must be chosen.
option(PAGMO_BUILD_PAGMO "Build pagmo." ON)
option(PAGMO_BUILD_PYGMO "Build pygmo." OFF)

# Check consistency.
if(PAGMO_BUILD_PAGMO AND PAGMO_BUILD_PYGMO)
    message(FATAL_ERROR "Please select whether to build pagmo or pygmo: you cannot build them both at the same time.")
endif()

if((NOT PAGMO_BUILD_PAGMO) AND (NOT PAGMO_BUILD_PYGMO))
    message(FATAL_ERROR "Please select if you want to build pagmo or pygmo.")
endif()

# Main pagmo/pygmo project version.
set(PAGMO_PROJECT_VERSION 2.11)

if(PAGMO_BUILD_PAGMO)
    # Initial setup of a pagmo build.
    project(pagmo VERSION ${PAGMO_PROJECT_VERSION} LANGUAGES CXX C)
    message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
    enable_testing()

    # Build option: enable test set.
    option(PAGMO_BUILD_TESTS "Build test set." OFF)

    # Build option: enable tutorials.
    option(PAGMO_BUILD_TUTORIALS "Build tutorials." OFF)

    # Build option: enable features depending on Eigen3.
    option(PAGMO_WITH_EIGEN3 "Enable features depending on Eigen3 (such as CMAES). Requires Eigen3." OFF)

    # Build option: enable NLopt.
    option(PAGMO_WITH_NLOPT "Enable wrappers for the NLopt algorithms." OFF)

    # Build option: enable Ipopt.
    option(PAGMO_WITH_IPOPT "Enable wrappers for the Ipopt solver." OFF)

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
else()
    # Initial setup of a pygmo build.
    project(pygmo VERSION ${PAGMO_PROJECT_VERSION} LANGUAGES CXX C)
    message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
endif()

# Common general bits.

# Initial setup of compiler flags.
include(YACMACompilerLinkerSettings)

# Threading setup.
include(YACMAThreadingSetup)

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
  # and enable the bigobj option.
  list(APPEND PAGMO_CXX_FLAGS_DEBUG "-DNOMINMAX" "/wd4459" "/wd4127" "/wd4702" "/bigobj")
  list(APPEND PAGMO_CXX_FLAGS_RELEASE "-DNOMINMAX" "/wd4459" "/wd4127" "/wd4702" "/bigobj")
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

if(PAGMO_BUILD_PAGMO)
    # pagmo dependencies.

    # TBB.
    find_package(TBB REQUIRED)

    # Eigen3
    if(PAGMO_WITH_EIGEN3)
        find_package(Eigen3 REQUIRED)
        message(STATUS "Eigen include directory: ${EIGEN3_INCLUDE_DIR}")
        message(STATUS "Eigen version detected: ${EIGEN3_VERSION}")
    endif()

    # NLopt
    if(PAGMO_WITH_NLOPT)
        find_package(NLOPT REQUIRED)
        message(STATUS "NLopt include directory: ${NLOPT_INCLUDE_DIR}")
        message(STATUS "NLopt library: ${NLOPT_LIBRARY}")
    endif()

    # Ipopt
    if(PAGMO_WITH_IPOPT)
        find_package(IPOPT REQUIRED)
        message(STATUS "Ipopt include directory: ${IPOPT_INCLUDE_DIR}")
        message(STATUS "Ipopt library: ${IPOPT_LIBRARY}")
    endif()
endif()

if(PAGMO_BUILD_PYGMO)
    # pygmo dependencies.
    include(YACMAPythonSetup)

    # Python version check.
    if(${PYTHON_VERSION_MAJOR} LESS 2 OR (${PYTHON_VERSION_MAJOR} EQUAL 2 AND ${PYTHON_VERSION_MINOR} LESS 7))
        message(FATAL_ERROR "Minimum supported Python version is 2.7.")
    endif()

    # NOTE: for the time being, require that pagmo/pygmo versions are matching exactly.
    find_package(Pagmo ${PAGMO_PROJECT_VERSION} EXACT REQUIRED)

    find_package(NumPy REQUIRED)

    # Internal variable that will be used to tell PagmoFindBoost to locate Boost.Python.
    set(_PAGMO_FIND_BOOST_PYTHON TRUE)
endif()

# Boost setup (common to pagmo/pygmo).
include(PagmoFindBoost)

if(PAGMO_BUILD_PAGMO)
    # List of source files.
    set(PAGMO_SRC_FILES
        # Core classes.
        "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/population.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/problem.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/bfe.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/island.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/archipelago.cpp"
        # UDP.
        "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2006.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2009.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/schwefel.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/rosenbrock.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/hock_schittkowsky_71.cpp"
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
        # UDA.
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
        # UDI.
        "${CMAKE_CURRENT_SOURCE_DIR}/src/islands/thread_island.cpp"
        # UDBFE.
        "${CMAKE_CURRENT_SOURCE_DIR}/src/batch_evaluators/default_bfe.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/batch_evaluators/member_bfe.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/batch_evaluators/thread_bfe.cpp"
        # Utils.
        "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/constrained.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/discrepancy.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/generic.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/multi_objective.cpp"
        # Detail.
        "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/bfe_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/task_queue.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/prime_numbers.cpp"
    )

    if(NOT YACMA_COMPILER_IS_MSVC AND NOT APPLE)
        set(PAGMO_SRC_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2014.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2014_data.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2013.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cec2013_data.cpp"
            "${PAGMO_SRC_FILES}"
        )
    endif()

    if(PAGMO_WITH_FORK_ISLAND)
        set(PAGMO_SRC_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/islands/fork_island.cpp"
            "${PAGMO_SRC_FILES}"
        )
    endif()

    # Setup of the pagmo library.
    add_library(pagmo SHARED "${PAGMO_SRC_FILES}")
    set_property(TARGET pagmo PROPERTY VERSION "${pagmo_VERSION_MAJOR}.${pagmo_VERSION_MINOR}")
    set_property(TARGET pagmo PROPERTY SOVERSION 0)
    target_compile_options(pagmo PRIVATE "$<$<CONFIG:DEBUG>:${PAGMO_CXX_FLAGS_DEBUG}>" "$<$<CONFIG:RELEASE>:${PAGMO_CXX_FLAGS_RELEASE}>")
    set_target_properties(pagmo PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(pagmo PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)
    # NOTE: for clang-cl, cmake tries to set -std=c++11 here, which makes
    # it error out. Disable it as MSVC 2015 is implicitly C++14 anyway.
    if(NOT (YACMA_COMPILER_IS_MSVC AND YACMA_COMPILER_IS_CLANGXX))
        # Let's setup the target C++ standard, but only if the user did not provide it manually.
        if(NOT CMAKE_CXX_STANDARD)
            set_property(TARGET pagmo PROPERTY CXX_STANDARD 11)
        endif()
        set_property(TARGET pagmo PROPERTY CXX_STANDARD_REQUIRED YES)
        set_property(TARGET pagmo PROPERTY CXX_EXTENSIONS NO)
    endif()
    target_include_directories(pagmo PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>)

    # Threads.
    target_link_libraries(pagmo PUBLIC Threads::Threads)

    # Boost.
    target_link_libraries(pagmo PUBLIC Boost::boost)

    # TBB.
    target_link_libraries(pagmo PUBLIC TBB::tbb)

    # Boost serialization.
    target_link_libraries(pagmo PUBLIC Boost::serialization)

    if(PAGMO_WITH_EIGEN3)
        # Link pagmo to eigen3.
        target_link_libraries(pagmo PUBLIC Eigen3::eigen3)
        set(PAGMO_ENABLE_EIGEN3 "#define PAGMO_WITH_EIGEN3")
    endif()

    if(PAGMO_WITH_NLOPT)
        # Link pagmo to NLopt.
        target_link_libraries(pagmo PUBLIC NLOPT::nlopt)
        set(PAGMO_ENABLE_NLOPT "#define PAGMO_WITH_NLOPT")
    endif()

    if(PAGMO_WITH_IPOPT)
        # Link pagmo to Ipopt.
        target_link_libraries(pagmo PUBLIC IPOPT::ipopt)
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

    if(PAGMO_BUILD_TUTORIALS)
        add_subdirectory("${CMAKE_SOURCE_DIR}/tutorials")
    endif()
endif()

if(PAGMO_BUILD_PYGMO)
    # Add the pygmo subdirectory.
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

# Library installation.
if(PAGMO_BUILD_PAGMO)
    # Installation of the header files.
    install(DIRECTORY include/ DESTINATION include)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/pagmo/config.hpp" DESTINATION include/pagmo)

    # Installation of the library.
    install(TARGETS pagmo
        EXPORT pagmo_export
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )

    # Setup of the optional deps.
    set(_PAGMO_CONFIG_OPTIONAL_DEPS)
    if(PAGMO_WITH_EIGEN3)
        set(_PAGMO_CONFIG_OPTIONAL_DEPS "${_PAGMO_CONFIG_OPTIONAL_DEPS}find_package(Eigen3 REQUIRED)\n")
    endif()
    if(PAGMO_WITH_NLOPT)
        set(_PAGMO_CONFIG_OPTIONAL_DEPS "${_PAGMO_CONFIG_OPTIONAL_DEPS}find_package(NLOPT REQUIRED)\n")
    endif()
    if(PAGMO_WITH_IPOPT)
        set(_PAGMO_CONFIG_OPTIONAL_DEPS "${_PAGMO_CONFIG_OPTIONAL_DEPS}find_package(IPOPT REQUIRED)\n")
    endif()

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pagmo-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/pagmo-config.cmake" @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pagmo-config.cmake" DESTINATION "lib/cmake/pagmo")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindEigen3.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindNLOPT.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindIPOPT.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/PagmoFindBoost.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindTBB.cmake" DESTINATION "lib/cmake/pagmo")
    install(EXPORT pagmo_export NAMESPACE Pagmo:: DESTINATION lib/cmake/pagmo)
    # Take care of versioning.
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/pagmo-config-version.cmake" VERSION ${pagmo_VERSION}
        COMPATIBILITY ExactVersion)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pagmo-config-version.cmake" DESTINATION "lib/cmake/pagmo")
endif()
