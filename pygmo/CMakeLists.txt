if(UNIX)
    # On unix platforms, we set the rpath of the pygmo libraries
    # to the lib subdirectory of the install prefix. This allows
    # to install pygmo in nonstandard locations (e.g., ~/.local)
    # without having to muck around with library search paths.
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# The name of the BP target to link to depends on the Boost and Python versions.
# NOTE: since Boost 1.67, the naming of the Boost.Python library has changed to include the
# major and minor python version as a suffix. See the release notes:
# https://www.boost.org/users/history/version_1_67_0.html
if(${Boost_MAJOR_VERSION} GREATER 1 OR (${Boost_MAJOR_VERSION} EQUAL 1 AND ${Boost_MINOR_VERSION} GREATER 66))
    set(PYGMO_BP_TARGET "Boost::python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}")
else()
    if(${PYTHON_VERSION_MAJOR} EQUAL 2)
        set(PYGMO_BP_TARGET "Boost::python")
    else()
        set(PYGMO_BP_TARGET "Boost::python3")
    endif()
endif()

# Setup of the config.hpp file for pygmo.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/include/pygmo/config.hpp" @ONLY)

# Setup of the header-only pygmo library.
add_library(pygmo INTERFACE)
target_link_libraries(pygmo INTERFACE Pagmo::pagmo ${PYGMO_BP_TARGET} Boost::disable_autolinking YACMA::PythonModule NumPy::numpy)
target_include_directories(pygmo INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/pygmo> $<INSTALL_INTERFACE:include>)
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
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/pygmo/config.hpp" DESTINATION include/pygmo)

# Setup of the pygmo core module.
YACMA_PYTHON_MODULE(core
    algorithm.cpp
    bfe.cpp
    topology.cpp
    island.cpp
    problem.cpp
    r_policy.cpp
    s_policy.cpp
    common_base.cpp
    object_serialization.cpp
    handle_thread_py_exception.cpp
    core.cpp
    docstrings.cpp
    expose_algorithms_0.cpp
    expose_algorithms_1.cpp
    expose_problems_0.cpp
    expose_problems_1.cpp
    expose_islands.cpp
    expose_bfes.cpp
    expose_topologies.cpp
    expose_r_policies.cpp
    expose_s_policies.cpp
)
target_link_libraries(core PRIVATE Pagmo::pagmo ${PYGMO_BP_TARGET} Boost::disable_autolinking NumPy::numpy pygmo)
target_compile_options(core PRIVATE "$<$<CONFIG:DEBUG>:${PAGMO_CXX_FLAGS_DEBUG}>" "$<$<CONFIG:RELEASE>:${PAGMO_CXX_FLAGS_RELEASE}>")
# NOTE: some versions of Boost before 1.66 have wrongly defined visibility settings, hence
# we enable default hidden visibility only for later versions.
if(${Boost_MAJOR_VERSION} GREATER 1 OR (${Boost_MAJOR_VERSION} EQUAL 1 AND ${Boost_MINOR_VERSION} GREATER 65))
    set_target_properties(core PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(core PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)
endif()
# Let's setup the target C++ standard, but only if the user did not provide it manually.
if(NOT CMAKE_CXX_STANDARD)
    set_property(TARGET core PROPERTY CXX_STANDARD 11)
endif()
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
install(FILES __init__.py test.py _patch_problem.py _patch_algorithm.py _patch_bfe.py _patch_island.py _patch_topology.py _patch_r_policy.py
     _patch_s_policy.py _problem_test.py _algorithm_test.py _island_test.py _topology_test.py _r_policy_test.py _s_policy_test.py
     _py_islands.py _py_problems.py _mp_utils.py
     "${CMAKE_CURRENT_BINARY_DIR}/_version.py" DESTINATION ${PYGMO_INSTALL_PATH})

# pygmo's public headers, to be installed.
set (PYGMO_HEADERS
    algorithm_exposition_suite.hpp
    common_utils.hpp
    function_traits.hpp
    island_exposition_suite.hpp
    numpy.hpp
    problem_exposition_suite.hpp
    python_includes.hpp
    register_ap.hpp
)

# Install the pygmo headers.
install(FILES ${PYGMO_HEADERS} DESTINATION include/pygmo)
