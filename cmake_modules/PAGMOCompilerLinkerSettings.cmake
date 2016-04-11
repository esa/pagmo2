include(YACMACompilerLinkerSettings)

# This is valid for GCC, clang and Intel. I think that MSVC has the std version hardcoded.
# TODO version checking for MSVC?
if(YACMA_COMPILER_IS_CLANGXX OR YACMA_COMPILER_IS_GNUCXX OR YACMA_COMPILER_IS_INTELXX)
    set(PAGMO_CHECK_CXX_FLAG)
    check_cxx_compiler_flag("-std=c++14" PAGMO_CHECK_CXX_FLAG)
    if(PAGMO_CHECK_CXX_FLAG)
        message(STATUS "C++14 supported by the compiler, enabling.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        message(FATAL_ERROR "C++14 is not supported by the compiler, aborting.")
    endif()
endif()
