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
if(APPLE)
    message(STATUS "OSX detected, setting the 'CMAKE_MACOSX_RPATH' option to TRUE.")
    set(CMAKE_MACOSX_RPATH TRUE)
endif()

# Helper function to print out the autodetected flags.
function(_YACMA_REPORT_FLAGS)
    message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "YACMA autodetected C++ flags: ${YACMA_CXX_FLAGS}")
    message(STATUS "YACMA autodetected C++ debug flags: ${YACMA_CXX_FLAGS_DEBUG}")
endfunction()

# Enable conditionally a CXX flag, if supported by the compiler.
# This is for flags intended to be enabled in all configurations.
# NOTE: we use macros because it's apparently impossible to append to an internal
# CACHEd list.
macro(_YACMA_CHECK_ENABLE_CXX_FLAG flag)
    set(CMAKE_REQUIRED_QUIET TRUE)
    check_cxx_compiler_flag("${flag}" YACMA_CHECK_CXX_FLAG::${flag})
    unset(CMAKE_REQUIRED_QUIET)
    if(YACMA_CHECK_CXX_FLAG::${flag})
        message(STATUS "'${flag}': flag is supported by the compiler, enabling.")
        list(APPEND _YACMA_CXX_FLAGS "${flag}")
    else()
        message(STATUS "'${flag}': flag is not supported by the compiler.")
    endif()
endmacro()

# Enable conditionally a debug CXX flag, is supported by the compiler.
# This is for flags intended to be enabled in debug mode.
macro(_YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG flag)
    set(CMAKE_REQUIRED_QUIET TRUE)
    check_cxx_compiler_flag("${flag}" YACMA_CHECK_DEBUG_CXX_FLAG::${flag})
    unset(CMAKE_REQUIRED_QUIET)
    if(YACMA_CHECK_DEBUG_CXX_FLAG::${flag})
        message(STATUS "'${flag}': debug flag is supported by the compiler, enabling.")
        list(APPEND _YACMA_CXX_FLAGS_DEBUG "${flag}")
    else()
        message(STATUS "'${flag}': debug flag is not supported by the compiler.")
    endif()
endmacro()

# What we want to avoid is to re-run the expensive flag checks. We will set cache variables
# on the initial run and skip following CMake runs.
if(NOT _YACMACompilerLinkerSettingsRun)
    # Init the flags lists.
    set(_YACMA_CXX_FLAGS "")
    set(_YACMA_CXX_FLAGS_DEBUG "")

    # Configuration bits specific for GCC.
    if(YACMA_COMPILER_IS_GNUCXX)
        _YACMA_CHECK_ENABLE_CXX_FLAG(-fdiagnostics-color=auto)
        # New in GCC 9.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Waddress-of-packed-member)
    endif()

    # Configuration bits specific for clang.
    if(YACMA_COMPILER_IS_CLANGXX)
        # For now it seems like -Wshadow from clang behaves better than GCC's, just enable it here
        # for the time being.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wshadow)
        # Clang is better at this flag than GCC.
        # NOTE: enable unconditionally, as it seems like the CMake
        # machinery for detecting this fails. Perhaps the source code
        # used for checking the flag emits warnings?
        list(APPEND _YACMA_CXX_FLAGS_DEBUG "-Werror")
        # New warnings in clang 8.
        # NOTE: a few issues with macros here, let's disable for now.
        # _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wextra-semi-stmt)
        # New warnings in clang 10.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wtautological-overlap-compare)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wtautological-compare)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wtautological-bitwise-compare)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wbitwise-conditional-parentheses)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wrange-loop-analysis)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wmisleading-indentation)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wc99-designator)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wreorder-init-list)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wsizeof-pointer-div)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wsizeof-array-div)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wxor-used-as-pow)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wfinal-dtor-non-final-class)
        # NOTE: this is a new flag in Clang 13 which seems to give
        # incorrect warnings for UDLs.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wno-reserved-identifier)
    endif()

    # Common configuration for GCC, clang and Intel.
    if(YACMA_COMPILER_IS_CLANGXX OR YACMA_COMPILER_IS_INTELXX OR YACMA_COMPILER_IS_GNUCXX)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wall)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wextra)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wnon-virtual-dtor)
        # NOTE: this flag is a bit too chatty, let's disable it for the moment.
        #_YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wnoexcept)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wlogical-op)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wconversion)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wdeprecated)
        # This limit is supposed to be at least 1024 in C++11, but for some reason
        # clang sets this to 256, and gcc to 900.
        _YACMA_CHECK_ENABLE_CXX_FLAG(-ftemplate-depth=1024)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wold-style-cast)
        # NOTE: disable this for now, as it results in a lot of clutter from Boost.
        # _YACMA_CHECK_ENABLE_CXX_FLAG(-Wzero-as-null-pointer-constant)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-pedantic-errors)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wdisabled-optimization)
        # This is useful when the compiler decides the template backtrace is too verbose.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-ftemplate-backtrace-limit=0)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-fstack-protector-all)
        # From GCC 5.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wodr)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wsuggest-final-types)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wsuggest-final-methods)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wsuggest-override)
        # From GCC 6.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wshift-negative-value)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wshift-overflow=2)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wduplicated-cond)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wnull-dereference)
        # From GCC 7.
        #_YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wduplicated-branches)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wrestrict)
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Waligned-new)
        # From GCC 8.
        _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wcast-align=strict)
        # This is supposed to produce a nice graphical visualization
        # of mismatching template errors.
        _YACMA_CHECK_ENABLE_CXX_FLAG(-fdiagnostics-show-template-tree)
        if(YACMA_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "5.999")
            # NOTE: GCC >= 6 seems to be wrongly warning about visibility attributes
            # in some situations:
            # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
            # Let's just disable the warning for now.
            message(STATUS "Activating the '-Wno-attributes' workaround for GCC >= 6.")
            _YACMA_CHECK_ENABLE_CXX_FLAG(-Wno-attributes)
        endif()
        if(YACMA_COMPILER_IS_GNUCXX)
            # The -Wmaybe-uninitialized flag is enabled by -Wall, but it is known
            # to emit a lot of possibly spurious warnings. Let's just disable it.
            message(STATUS "Activating the '-Wno-maybe-uninitialized' workaround for GCC.")
            _YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wno-maybe-uninitialized)
        endif()
    endif()

    # MSVC setup.
    if(YACMA_COMPILER_IS_MSVC AND NOT YACMA_COMPILER_IS_CLANGXX)
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
