if(YACMACompilerLinkerSettingsIncluded)
    return()
endif()

include(CheckCXXCompilerFlag)

message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")

# Clang detection:
# http://stackoverflow.com/questions/10046114/in-cmake-how-can-i-test-if-the-compiler-is-clang
# http://www.cmake.org/cmake/help/v2.8.10/cmake.html#variable:CMAKE_LANG_COMPILER_ID
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(YACMA_COMPILER_IS_CLANGXX 1)
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(YACMA_COMPILER_IS_INTELXX 1)
endif()

if(MSVC)
    set(YACMA_COMPILER_IS_MSVC 1)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(YACMA_COMPILER_IS_GNUCXX 1)
endif()

# Enable conditionally a CXX flags, is supported by the compiler.
# The flag will be enabled in all buld types.
macro(YACMA_CHECK_ENABLE_CXX_FLAG flag)
    set(YACMA_CHECK_CXX_FLAG)
    check_cxx_compiler_flag("${flag}" YACMA_CHECK_CXX_FLAG)
    if(YACMA_CHECK_CXX_FLAG)
        message(STATUS "Enabling the '${flag}' compiler flag.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
    else()
        message(STATUS "Disabling the '${flag}' compiler flag.")
    endif()
    unset(YACMA_CHECK_CXX_FLAG CACHE)
endmacro()

# Enable conditionally a debug CXX flags, is supported by the compiler.
# The flag will be enabled only in debug builds.
macro(YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG flag)
    set(YACMA_CHECK_DEBUG_CXX_FLAG)
    check_cxx_compiler_flag("${flag}" YACMA_CHECK_DEBUG_CXX_FLAG)
    if(YACMA_CHECK_DEBUG_CXX_FLAG)
        message(STATUS "Enabling the '${flag}' debug compiler flag.")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${flag}")
    else()
        message(STATUS "Disabling the '${flag}' debug compiler flag.")
    endif()
    unset(YACMA_CHECK_DEBUG_CXX_FLAG CACHE)
endmacro()

# NOTE: all these flags are with a Unix-like syntax. We will need to change them
# for MSVC and clang on windows possibly.
macro(YACMA_SETUP_CXX_FLAGS)
    # Configuration bits specific for GCC.
    if(YACMA_COMPILER_IS_GNUCXX)
        YACMA_CHECK_ENABLE_CXX_FLAG(-fdiagnostics-color=auto)
    endif()

    # Configuration bits specific for clang.
    if(YACMA_COMPILER_IS_CLANGXX)
        # For now it seems like -Wshadow from clang behaves better than GCC's, just enable it here
        # for the time being.
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wshadow)
        # Clang is better at this flag than GCC.
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Werror)
    endif()

    # Common configuration for GCC, clang and Intel.
    if (YACMA_COMPILER_IS_CLANGXX OR YACMA_COMPILER_IS_INTELXX OR YACMA_COMPILER_IS_GNUCXX)
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wall)
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wextra)
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wnon-virtual-dtor)
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wnoexcept)
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wlogical-op)
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wconversion)
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wdeprecated)
        # This limit is supposed to be at least 1024 in C++11, but for some reason
        # clang sets this to 256, and gcc to 900.
        YACMA_CHECK_ENABLE_CXX_FLAG(-ftemplate-depth=1024)
        # NOTE: this can be useful, but at the moment it triggers lots of warnings in type traits.
        # Keep it in mind for the next time we touch type traits.
        # YACMA_CHECK_ENABLE_CXX_FLAG(-Wold-style-cast)
        # NOTE: disable this for now, as it results in a lot of clutter from Boost.
        # YACMA_CHECK_ENABLE_CXX_FLAG(-Wzero-as-null-pointer-constant)
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-pedantic-errors)
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wdisabled-optimization)
        YACMA_CHECK_ENABLE_CXX_FLAG(-fvisibility-inlines-hidden)
        YACMA_CHECK_ENABLE_CXX_FLAG(-fvisibility=hidden)
        # This is useful when the compiler decides the template backtrace is too verbose.
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-ftemplate-backtrace-limit=0)
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-fstack-protector-all)
        # This became available in GCC at one point.
        YACMA_CHECK_ENABLE_DEBUG_CXX_FLAG(-Wodr)
    endif()
endmacro()

# This is an OS X specific setting that is suggested to be enabled. See:
# https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/
# http://stackoverflow.com/questions/31561309/cmake-warnings-under-os-x-macosx-rpath-is-not-specified-for-the-following-targe
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
endif()

# Mark as included.
set(YACMACompilerLinkerSettingsIncluded YES)
