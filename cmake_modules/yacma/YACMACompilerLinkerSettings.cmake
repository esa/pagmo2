if(_YACMACompilerLinkerSettingsIncluded)
    return()
endif()

include(CheckCXXCompilerFlag)

# NOTE: we want to make sure the following variables are defined each time we include
# this file, even when the file is re-included (e.g., from a parallel unrelated tree).

# Clang detection:
# http://stackoverflow.com/questions/10046114/in-cmake-how-can-i-test-if-the-compiler-is-clang
# http://www.cmake.org/cmake/help/v2.8.10/cmake.html#variable:CMAKE_LANG_COMPILER_ID
# NOTE: we use MATCHES here because on OSX sometimes the compiler calls itself "AppleClang".
if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    set(YACMA_COMPILER_IS_CLANGXX TRUE)
endif()

if(${CMAKE_CXX_COMPILER_ID} MATCHES "Intel")
    set(YACMA_COMPILER_IS_INTELXX TRUE)
endif()

if(MSVC)
    set(YACMA_COMPILER_IS_MSVC TRUE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(YACMA_COMPILER_IS_GNUCXX TRUE)
endif()

# This is an OS X specific setting that is suggested to be enabled. See:
# https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/
# http://stackoverflow.com/questions/31561309/cmake-warnings-under-os-x-macosx-rpath-is-not-specified-for-the-following-targe
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH TRUE)
endif()

# Helper function to print out the autodetected flags.
function(_YACMA_REPORT_FLAGS)
    message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "YACMA autodetected C++ flags: ${YACMA_CXX_FLAGS}")
    message(STATUS "YACMA autodetected C++ debug flags: ${YACMA_CXX_FLAGS_DEBUG}")
endfunction()

# Enable conditionally a CXX flags, if supported by the compiler.
# This is for flags intended to be enabled in all configurations.
# NOTE: we use macros and go through temporary private variables
# because it's apparently impossible to append to an internal
# CACHEd list.
macro(_YACMA_CHECK_ENABLE_CXX_FLAG flag)
    set(CMAKE_REQUIRED_QUIET TRUE)
    check_cxx_compiler_flag("${flag}" YACMA_CHECK_CXX_FLAG)
    unset(CMAKE_REQUIRED_QUIET)
    if(YACMA_CHECK_CXX_FLAG)
        message(STATUS "'${flag}': flag is supported by the compiler, enabling.")
        list(APPEND _YACMA_CXX_FLAGS "${flag}")
    else()
        message(STATUS "'${flag}': flag is not supported by the compiler.")
    endif()
    # NOTE: check_cxx_compiler stores variables in the cache.
    unset(YACMA_CHECK_CXX_FLAG CACHE)
endmacro()

# Enable conditionally a debug CXX flags, is supported by the compiler.
# This is for flags intended to be enabled in debug mode.
macro(_YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG flag)
    set(CMAKE_REQUIRED_QUIET TRUE)
    check_cxx_compiler_flag("${flag}" YACMA_CHECK_DEBUG_CXX_FLAG)
    unset(CMAKE_REQUIRED_QUIET)
    if(YACMA_CHECK_DEBUG_CXX_FLAG)
        message(STATUS "'${flag}': debug flag is supported by the compiler, enabling.")
        list(APPEND _YACMA_CXX_FLAGS_DEBUG "${flag}")
    else()
        message(STATUS "'${flag}': debug flag is not supported by the compiler.")
    endif()
    unset(YACMA_CHECK_DEBUG_CXX_FLAG CACHE)
endmacro()

# What we want to avoid is to re-run the expensive flag checks. We will set cache variables
# on the initial run and skip following CMake runs.
if(NOT _YACMACompilerLinkerSettingsRun)
    # Init the flags lists.
    set(_YACMA_CXX_FLAGS "")
    set(_YACMA_CXX_FLAGS_DEBUG "")

    # NOTE: all these flags are with a Unix-like syntax. We will need to change them
    # for MSVC and clang on windows possibly.

    # Configuration bits specific for GCC.
    if(YACMA_COMPILER_IS_GNUCXX)
        _YACMA_CHECK_ENABLE_CXX_FLAG(-fdiagnostics-color=auto)
    endif()

    # Configuration bits specific for clang.
    if(YACMA_COMPILER_IS_CLANGXX AND NOT YACMA_COMPILER_IS_MSVC)
        # For now it seems like -Wshadow from clang behaves better than GCC's, just enable it here
        # for the time being.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wshadow)
        # Clang is better at this flag than GCC.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Werror)
    endif()

    # Common configuration for GCC, clang and Intel.
    if ((YACMA_COMPILER_IS_CLANGXX AND NOT YACMA_COMPILER_IS_MSVC) OR YACMA_COMPILER_IS_INTELXX OR YACMA_COMPILER_IS_GNUCXX)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wall)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wextra)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wnon-virtual-dtor)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wnoexcept)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wlogical-op)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wconversion)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wdeprecated)
        # This limit is supposed to be at least 1024 in C++11, but for some reason
        # clang sets this to 256, and gcc to 900.
        _YACMA_CHECK_ENABLE_CXX_FLAG(-ftemplate-depth=1024)
        # NOTE: this can be useful, but at the moment it triggers lots of warnings in type traits.
        # Keep it in mind for the next time we touch type traits.
        # _YACMA_CHECK_ENABLE_CXX_FLAG(-Wold-style-cast)
        # NOTE: disable this for now, as it results in a lot of clutter from Boost.
        # _YACMA_CHECK_ENABLE_CXX_FLAG(-Wzero-as-null-pointer-constant)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-pedantic-errors)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wdisabled-optimization)
        _YACMA_CHECK_ENABLE_CXX_FLAG(-fvisibility-inlines-hidden)
        _YACMA_CHECK_ENABLE_CXX_FLAG(-fvisibility=hidden)
        # This is useful when the compiler decides the template backtrace is too verbose.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-ftemplate-backtrace-limit=0)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-fstack-protector-all)
        # These became available in GCC from version 5.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wodr)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wsuggest-final-types)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wsuggest-final-methods)
    endif()

    # MSVC setup.
    if(YACMA_COMPILER_IS_MSVC)
        # Enable higher warning level than usual.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(/W4)
        # Treat warnings as errors.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(/WX)
    endif()

    # Set the cache variables.
    set(YACMA_CXX_FLAGS "${_YACMA_CXX_FLAGS}" CACHE INTERNAL "")
    set(YACMA_CXX_FLAGS_DEBUG "${_YACMA_CXX_FLAGS_DEBUG}" CACHE INTERNAL "")
    set(_YACMACompilerLinkerSettingsRun YES CACHE INTERNAL "")
endif()

# Final report.
_YACMA_REPORT_FLAGS()

# Mark as included.
set(_YACMACompilerLinkerSettingsIncluded YES)
