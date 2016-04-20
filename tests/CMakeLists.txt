# NOTE: this is hard-coded at the moment, ideally we would want to detect this
# in the build system. This needs to be set only if the Boost libraries are dynamic.
add_definitions(-DBOOST_TEST_DYN_LINK)

# Include boost directories.
include_directories(${Boost_INCLUDE_DIRS})

macro(ADD_PAGMO_TESTCASE arg1)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_executable(${arg1} ${arg1}.cpp)
        target_link_libraries(${arg1} ${MANDATORY_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
        add_test(${arg1} ${arg1})
    endif()
endmacro()

ADD_PAGMO_TESTCASE(io)
ADD_PAGMO_TESTCASE(problem)
ADD_PAGMO_TESTCASE(rng)
ADD_PAGMO_TESTCASE(type_traits)

