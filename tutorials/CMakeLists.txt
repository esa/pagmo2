function(ADD_PAGMO_TUTORIAL arg1)
    add_executable(${arg1} ${arg1}.cpp)
    target_link_libraries(${arg1} PRIVATE pagmo)
    target_compile_options(${arg1} PRIVATE
        "$<$<CONFIG:Debug>:${PAGMO_CXX_FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${PAGMO_CXX_FLAGS_RELEASE}>"
        "$<$<CONFIG:RelWithDebInfo>:${PAGMO_CXX_FLAGS_RELEASE}>"
        "$<$<CONFIG:MinSizeRel>:${PAGMO_CXX_FLAGS_RELEASE}>"
    )
    # Set the minimum C++ standard to C++17
    target_compile_features(${arg1} PRIVATE cxx_std_17)
    set_property(TARGET ${arg1} PROPERTY CXX_EXTENSIONS NO)
    target_compile_definitions(${arg1} PRIVATE BOOST_ALLOW_DEPRECATED_HEADERS)
    add_test(${arg1} ${arg1})
endfunction()

ADD_PAGMO_TUTORIAL(getting_started)
ADD_PAGMO_TUTORIAL(first_udp_ver0)
ADD_PAGMO_TUTORIAL(first_udp_ver1)
ADD_PAGMO_TUTORIAL(nsga2_example)
ADD_PAGMO_TUTORIAL(problem_basic)
ADD_PAGMO_TUTORIAL(problem_basic_gh)
ADD_PAGMO_TUTORIAL(problem_basic_s)
