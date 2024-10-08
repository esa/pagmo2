# We move the test files into the testing directory
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/hypervolume_test_data/" DESTINATION "${CMAKE_BINARY_DIR}/tests/hypervolume_test_data/")

function(ADD_PAGMO_TESTCASE arg1)
    if(PAGMO_TEST_NSPLIT)
        math(EXPR __PAGMO_TEST_NUM "(${_PAGMO_TEST_NUM} + 1) % ${PAGMO_TEST_NSPLIT}")
        set(_PAGMO_TEST_NUM ${__PAGMO_TEST_NUM} PARENT_SCOPE)
    endif()
    if(PAGMO_TEST_NSPLIT AND "${PAGMO_TEST_SPLIT_NUM}" STREQUAL "${_PAGMO_TEST_NUM}")
        return()
    endif()
    add_executable(${arg1} ${arg1}.cpp)
    target_link_libraries(${arg1} PRIVATE pagmo PRIVATE Boost::unit_test_framework)
    target_compile_options(${arg1} PRIVATE
        "$<$<CONFIG:Debug>:${PAGMO_CXX_FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${PAGMO_CXX_FLAGS_RELEASE}>"
        "$<$<CONFIG:RelWithDebInfo>:${PAGMO_CXX_FLAGS_RELEASE}>"
        "$<$<CONFIG:MinSizeRel>:${PAGMO_CXX_FLAGS_RELEASE}>"
    )
    # Set the minimum C++ standard to C++17
    target_compile_features(${arg1} PRIVATE cxx_std_17)
    set_property(TARGET ${arg1} PROPERTY CXX_EXTENSIONS NO)
    set_property(TARGET ${arg1} PROPERTY UNITY_BUILD OFF)
    target_compile_definitions(${arg1} PRIVATE BOOST_ALLOW_DEPRECATED_HEADERS)
    add_test(NAME ${arg1} COMMAND ${arg1})
endfunction()

# Tests requiring no dependencies (in alphabetical order)
ADD_PAGMO_TESTCASE(ackley)
ADD_PAGMO_TESTCASE(algorithm)
ADD_PAGMO_TESTCASE(algorithm_type_traits)
ADD_PAGMO_TESTCASE(archipelago)
ADD_PAGMO_TESTCASE(archipelago_torture_test)
ADD_PAGMO_TESTCASE(base_bgl_topology)
ADD_PAGMO_TESTCASE(base_sr_policy)
ADD_PAGMO_TESTCASE(bfe)
ADD_PAGMO_TESTCASE(bee_colony)
ADD_PAGMO_TESTCASE(cec2006)
ADD_PAGMO_TESTCASE(cec2009)
ADD_PAGMO_TESTCASE(cec2013)
ADD_PAGMO_TESTCASE(cec2014)
ADD_PAGMO_TESTCASE(compass_search)
ADD_PAGMO_TESTCASE(constrained)
ADD_PAGMO_TESTCASE(custom_comparisons)
ADD_PAGMO_TESTCASE(cstrs_self_adaptive)
ADD_PAGMO_TESTCASE(de)
ADD_PAGMO_TESTCASE(de1220)
ADD_PAGMO_TESTCASE(decompose)
ADD_PAGMO_TESTCASE(default_bfe)
ADD_PAGMO_TESTCASE(discrepancy)
ADD_PAGMO_TESTCASE(dtlz)
ADD_PAGMO_TESTCASE(fair_replace)
ADD_PAGMO_TESTCASE(free_form)
ADD_PAGMO_TESTCASE(fully_connected)
ADD_PAGMO_TESTCASE(gwo)
ADD_PAGMO_TESTCASE(gaco)
ADD_PAGMO_TESTCASE(genetic_operators)
ADD_PAGMO_TESTCASE(generic)
ADD_PAGMO_TESTCASE(golomb_ruler)
ADD_PAGMO_TESTCASE(gradients_and_hessians)
ADD_PAGMO_TESTCASE(griewank)
ADD_PAGMO_TESTCASE(hypervolume)
ADD_PAGMO_TESTCASE(hock_schittkowski_71)
ADD_PAGMO_TESTCASE(inventory)
ADD_PAGMO_TESTCASE(lennard_jones)
ADD_PAGMO_TESTCASE(migration_torture_test)
ADD_PAGMO_TESTCASE(minlp_rastrigin)
ADD_PAGMO_TESTCASE(ihs)
ADD_PAGMO_TESTCASE(io)
ADD_PAGMO_TESTCASE(island)
ADD_PAGMO_TESTCASE(island_torture)
ADD_PAGMO_TESTCASE(luksan_vlcek1)
ADD_PAGMO_TESTCASE(maco)
ADD_PAGMO_TESTCASE(mbh)
ADD_PAGMO_TESTCASE(member_bfe)
ADD_PAGMO_TESTCASE(moead)
ADD_PAGMO_TESTCASE(moead_gen)
ADD_PAGMO_TESTCASE(multi_objective)
ADD_PAGMO_TESTCASE(nsga2)
ADD_PAGMO_TESTCASE(nspso)
ADD_PAGMO_TESTCASE(population)
ADD_PAGMO_TESTCASE(problem)
ADD_PAGMO_TESTCASE(problem_type_traits)
ADD_PAGMO_TESTCASE(pso)
ADD_PAGMO_TESTCASE(pso_gen)
ADD_PAGMO_TESTCASE(r_policy)
ADD_PAGMO_TESTCASE(rastrigin)
ADD_PAGMO_TESTCASE(ring)
ADD_PAGMO_TESTCASE(rng)
ADD_PAGMO_TESTCASE(rng_serialization)
ADD_PAGMO_TESTCASE(rosenbrock)
ADD_PAGMO_TESTCASE(s_policy)
ADD_PAGMO_TESTCASE(sade)
ADD_PAGMO_TESTCASE(simulated_annealing)
ADD_PAGMO_TESTCASE(sga)
ADD_PAGMO_TESTCASE(schwefel)
ADD_PAGMO_TESTCASE(sea)
ADD_PAGMO_TESTCASE(select_best)
ADD_PAGMO_TESTCASE(threading)
ADD_PAGMO_TESTCASE(thread_bfe)
ADD_PAGMO_TESTCASE(thread_island)
ADD_PAGMO_TESTCASE(topology)
ADD_PAGMO_TESTCASE(translate)
ADD_PAGMO_TESTCASE(type_traits)
ADD_PAGMO_TESTCASE(unconnected)
ADD_PAGMO_TESTCASE(unconstrain)
ADD_PAGMO_TESTCASE(wfg)
ADD_PAGMO_TESTCASE(zdt)

if(PAGMO_WITH_EIGEN3)
    ADD_PAGMO_TESTCASE(cmaes)
    ADD_PAGMO_TESTCASE(eigen3_serialization)
    ADD_PAGMO_TESTCASE(xnes)
endif()

if(PAGMO_WITH_NLOPT)
    ADD_PAGMO_TESTCASE(nlopt)
endif()

if(PAGMO_WITH_IPOPT)
    ADD_PAGMO_TESTCASE(ipopt)
endif()

if (PAGMO_WITH_FORK_ISLAND)
    ADD_PAGMO_TESTCASE(fork_island)
endif()
