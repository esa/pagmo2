macro(ADD_PAGMO_TUTORIAL arg1)
    add_executable(${arg1} ${arg1}.cpp)
    target_link_libraries(${arg1} ${MANDATORY_LIBRARIES})
    add_test(${arg1} ${arg1})
endmacro()

ADD_PAGMO_TUTORIAL(example0)
