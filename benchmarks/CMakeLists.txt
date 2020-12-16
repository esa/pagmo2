function(ADD_PAGMO_BENCHMARK arg1)
    add_executable(${arg1} ${arg1}.cpp)
    target_link_libraries(${arg1} PRIVATE pagmo)
    target_compile_options(${arg1} PRIVATE
        "$<$<CONFIG:Debug>:${PAGMO_CXX_FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${PAGMO_CXX_FLAGS_RELEASE}>"
        "$<$<CONFIG:RelWithDebInfo>:${PAGMO_CXX_FLAGS_RELEASE}>"
        "$<$<CONFIG:MinSizeRel>:${PAGMO_CXX_FLAGS_RELEASE}>"
    )
    # Set the minimum C++ standard to C++17
    target_compile_features(${arg1} PUBLIC cxx_std_17)
endfunction()

ADD_PAGMO_BENCHMARK(thread_island_pool)
