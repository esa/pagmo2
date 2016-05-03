if(YACMAPythonSetupIncluded)
    return()
endif()

# Need this to detect compiler.
include(YACMACompilerLinkerSettings)

# Find Python interpreter and libraries. This is the order suggested by CMake's documentation.
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
message(STATUS "Python interpreter: ${PYTHON_EXECUTABLE}")
message(STATUS "Python libraries: ${PYTHON_LIBRARIES}")
message(STATUS "Python include dirs: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Python library version: " ${PYTHONLIBS_VERSION_STRING})

# Include the Python header dirs.
include_directories(${PYTHON_INCLUDE_DIRS})

# This flag is used to signal the need to override the default extension of the Python modules
# depending on the architecture. Under Windows, for instance, CMake produces shared objects as
# .dll files, but Python from 2.5 onwards requires .pyd files (hence the need to override).
set(YACMA_PY_MODULE_EXTENSION "")

# Determine the installation path for Python modules.
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(STATUS "OS X platform detected.")
    # Apparently on OS X Python expects the .so extension for compiled modules.
	message(STATUS "Output extension for compiled modules will be '.so'.")
    set(YACMA_PY_MODULE_EXTENSION "so")
    # TODO fill in.
elseif(UNIX)
    message(STATUS "Generic UNIX platform detected.")
    # We must establish if the installation dir for Python modules is named 'site-packages' (as usual)
    # or 'dist-packages' (apparently Ubuntu 9.04 or maybe Python 2.6, it's not clear).
    execute_process(COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_LIST_DIR}/yacma_python_packages_dir.py"
        OUTPUT_VARIABLE _YACMA_PY_PACKAGES_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Python packages dir is: ${_YACMA_PY_PACKAGES_DIR}")
    set(YACMA_PYTHON_MODULES_INSTALL_PATH "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/${_YACMA_PY_PACKAGES_DIR}")
elseif(WIN32)
    message(STATUS "Windows platform detected.")
    message(STATUS "Output extension for compiled modules will be '.pyd'.")
    set(YACMA_PY_MODULE_EXTENSION "pyd")
    # TODO fill in.
endif()

message(STATUS "Python modules install path: " "${YACMA_PYTHON_MODULES_INSTALL_PATH}")

macro(YACMA_PYTHON_MODULE name)
    message(STATUS "Setting up the compilation of the Python module \"${name}\".")
    # A Python module is a shared library.
    add_library("${name}" SHARED ${ARGN})
    # Any "lib" prefix normally added by CMake must be removed.
    set_target_properties("${name}" PROPERTIES PREFIX "")
    if(NOT ${YACMA_PY_MODULE_EXTENSION} STREQUAL "")
        # If needed, set a custom extension for the module.
        message(STATUS "Setting up custom extension \"${YACMA_PY_MODULE_EXTENSION}\" for the Python module \"${name}\".")
        set_target_properties("${name}" PROPERTIES SUFFIX ".${YACMA_PY_MODULE_EXTENSION}")
    endif()
    # We need extra flags to be set when compiling Python modules, at least
    # with clang and gcc. See:
    # https://bugs.python.org/issue11149
    # http://www.python.org/dev/peps/pep-3123/
    # NOTE: not sure here how we should set flags up for MSVC or clang on windows, need
    # to check in the future.
    if(YACMA_COMPILER_IS_GNUCXX OR YACMA_COMPILER_IS_CLANGXX)
        message(STATUS "Setting up extra compiler flag \"-fwrapv\" for the Python module \"${name}\".")
        set_target_properties("${name}" PROPERTIES COMPILE_FLAGS "-fwrapv")
        set_target_properties("${name}" PROPERTIES LINK_FLAGS "-fwrapv")
        if(${PYTHON_VERSION_MAJOR} LESS 3)
            message(STATUS "Python < 3 detected, setting up extra compiler flag \"-fno-strict-aliasing\" for the Python module \"${name}.\"")
            set_target_properties("${name}" PROPERTIES COMPILE_FLAGS "-fno-strict-aliasing")
            set_target_properties("${name}" PROPERTIES LINK_FLAGS "-fno-strict-aliasing")
        endif()
    endif()
endmacro()

# Mark as included.
set(YACMAPythonSetupIncluded YES)
