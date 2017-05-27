if(UNIX)
    # On unix platforms, we set the rpath of the pygmo libraries
    # to the lib subdirectory of the install prefix. This allows
    # to install pygmo in nonstandard locations (e.g., ~/.local)
    # without having to muck around with library search paths.
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# The name of the BP target to link to depends on the Python version.
if(${PYTHON_VERSION_MAJOR} EQUAL 2)
    set(PYGMO_BP_TARGET "Boost::python")
else()
    set(PYGMO_BP_TARGET "Boost::python3")
endif()

# Boost.Python version 1.64 (and possibly later?) has issues with default hidden visibility.
if(${Boost_MAJOR_VERSION} GREATER 1 OR (${Boost_MAJOR_VERSION} EQUAL 1 AND ${Boost_MINOR_VERSION} GREATER 63))
    list(FIND PAGMO_CXX_FLAGS_DEBUG "-fvisibility=hidden" _PYGMO_HAS_VHIDDEN)
    if (NOT ${_PYGMO_HAS_VHIDDEN} EQUAL -1)
        message(STATUS "Removing the '-fvisibility=hidden' flag for Boost.Python > 1.63.")
        list(REMOVE_ITEM PAGMO_CXX_FLAGS_DEBUG "-fvisibility=hidden")
        list(REMOVE_ITEM PAGMO_CXX_FLAGS_RELEASE "-fvisibility=hidden")
    endif()
endif()

# The pygmo shared library.
add_library(pygmo SHARED pygmo.cpp)
target_link_libraries(pygmo PUBLIC ${PYGMO_BP_TARGET} Boost::disable_autolinking YACMA::PythonModule NumPy::numpy Pagmo::pagmo)
if(YACMA_COMPILER_IS_GNUCXX OR YACMA_COMPILER_IS_CLANGXX)
    # NOTE: this is a slight repetition of what is in yacma already.
    target_compile_options(pygmo PRIVATE "-fwrapv")
    if(${PYTHON_VERSION_MAJOR} LESS 3)
        target_compile_options(pygmo PRIVATE "-fno-strict-aliasing")
    endif()
endif()
target_compile_options(pygmo PRIVATE "$<$<CONFIG:DEBUG>:${PAGMO_CXX_FLAGS_DEBUG}>" "$<$<CONFIG:RELEASE>:${PAGMO_CXX_FLAGS_RELEASE}>")
set_property(TARGET pygmo PROPERTY CXX_STANDARD 11)
set_property(TARGET pygmo PROPERTY CXX_STANDARD_REQUIRED YES)
set_property(TARGET pygmo PROPERTY CXX_EXTENSIONS NO)
target_include_directories(pygmo PUBLIC $<INSTALL_INTERFACE:include>)
# Setup of the export.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pygmo-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/pygmo-config.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pygmo-config.cmake" DESTINATION "lib/cmake/pygmo")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules/FindNumPy.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules/PagmoFindBoost.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules/yacma/YACMAPythonSetup.cmake" DESTINATION "lib/cmake/pygmo")
# Take care of versioning.
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/pygmo-config-version.cmake" VERSION ${pygmo_VERSION}
    COMPATIBILITY ExactVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pygmo-config-version.cmake" DESTINATION "lib/cmake/pygmo")
install(TARGETS pygmo EXPORT pygmo_export ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(EXPORT pygmo_export NAMESPACE Pygmo:: DESTINATION lib/cmake/pygmo)

# Setup of the pygmo core module.
YACMA_PYTHON_MODULE(core core.cpp docstrings.cpp expose_algorithms.cpp expose_problems.cpp)
target_link_libraries(core PRIVATE ${PYGMO_BP_TARGET} Boost::disable_autolinking Pagmo::pagmo pygmo NumPy::numpy)
target_compile_options(core PRIVATE "$<$<CONFIG:DEBUG>:${PAGMO_CXX_FLAGS_DEBUG}>" "$<$<CONFIG:RELEASE>:${PAGMO_CXX_FLAGS_RELEASE}>")
set_property(TARGET core PROPERTY CXX_STANDARD 11)
set_property(TARGET core PROPERTY CXX_STANDARD_REQUIRED YES)
set_property(TARGET core PROPERTY CXX_EXTENSIONS NO)

# Helper file with version number.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in" "${CMAKE_CURRENT_BINARY_DIR}/_version.py" @ONLY)

# Setup the installation path.
set(PYGMO_INSTALL_PATH "${YACMA_PYTHON_MODULES_INSTALL_PATH}/pygmo")
install(TARGETS core
 RUNTIME DESTINATION ${PYGMO_INSTALL_PATH}
 LIBRARY DESTINATION ${PYGMO_INSTALL_PATH}
)

# Add submodules directories
add_subdirectory(plotting)

# Add the Python files.
install(FILES __init__.py test.py _patch_problem.py _patch_algorithm.py _problem_test.py
     _algorithm_test.py _island_test.py _py_islands.py "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
     DESTINATION ${PYGMO_INSTALL_PATH})

# pygmo's public headers, to be installed.
set (PYGMO_HEADERS
    algorithm_exposition_suite.hpp
    common_utils.hpp
    function_traits.hpp
    island_exposition_suite.hpp
    numpy.hpp
    problem_exposition_suite.hpp
    pygmo_classes.hpp
    python_includes.hpp
)

# Install the pygmo headers.
install(FILES ${PYGMO_HEADERS} DESTINATION include/pygmo)
