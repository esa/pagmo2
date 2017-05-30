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

# Setup of the header-only pygmo library.
add_library(pygmo INTERFACE)
target_link_libraries(pygmo INTERFACE ${PYGMO_BP_TARGET} Boost::disable_autolinking YACMA::PythonModule NumPy::numpy Pagmo::pagmo)
target_include_directories(pygmo INTERFACE $<INSTALL_INTERFACE:include>)
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
install(TARGETS pygmo EXPORT pygmo_export)
install(EXPORT pygmo_export NAMESPACE Pygmo:: DESTINATION lib/cmake/pygmo)

# Setup of the pygmo core module.
YACMA_PYTHON_MODULE(core core.cpp docstrings.cpp expose_algorithms.cpp expose_islands.cpp expose_problems.cpp)
target_link_libraries(core PRIVATE ${PYGMO_BP_TARGET} Boost::disable_autolinking Pagmo::pagmo NumPy::numpy)
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
    python_includes.hpp
)

# Install the pygmo headers.
install(FILES ${PYGMO_HEADERS} DESTINATION include/pygmo)
