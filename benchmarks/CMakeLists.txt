function(ADD_PAGMO_BENCHMARK arg1)
    add_executable(${arg1} ${arg1}.cpp)
    target_link_libraries(${arg1} PRIVATE pagmo)
    target_compile_options(${arg1} PRIVATE
        "$<$<CONFIG:Debug>:${PAGMO_CXX_FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${PAGMO_CXX_FLAGS_RELEASE}>"
        "$<$<CONFIG:RelWithDebInfo>:${PAGMO_CXX_FLAGS_RELEASE}>"
        "$<$<CONFIG:MinSizeRel>:${PAGMO_CXX_FLAGS_RELEASE}>"
    )
    # Let's setup the target C++ standard, but only if the user did not provide it manually.
    if(NOT CMAKE_CXX_STANDARD)
        set_property(TARGET ${arg1} PROPERTY CXX_STANDARD 17)
    endif()
    set_property(TARGET ${arg1} PROPERTY CXX_STANDARD_REQUIRED YES)
    set_property(TARGET ${arg1} PROPERTY CXX_EXTENSIONS NO)
endfunction()

ADD_PAGMO_BENCHMARK(thread_island_pool)
