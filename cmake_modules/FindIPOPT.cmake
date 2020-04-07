include(FindPackageHandleStandardArgs)

message(STATUS "Requested IPOPT components: ${IPOPT_FIND_COMPONENTS}")

# Check the components that were passed to find_package().
set(_IPOPT_ALLOWED_COMPONENTS header libipopt)
foreach(_IPOPT_CUR_COMPONENT ${IPOPT_FIND_COMPONENTS})
    if(NOT ${_IPOPT_CUR_COMPONENT} IN_LIST _IPOPT_ALLOWED_COMPONENTS)
        message(FATAL_ERROR "'${_IPOPT_CUR_COMPONENT}' is not a valid component for IPOPT.")
    endif()
endforeach()
unset(_IPOPT_ALLOWED_COMPONENTS)

# Setup the list of arguments to be passed to
# find_package_handle_standard_args().
set(_IPOPT_FPHSA_ARGS)

if ("header" IN_LIST IPOPT_FIND_COMPONENTS)
    # The header component was requested.
    # The associated variable is IPOPT_INCLUDE_DIR.
    list(APPEND _IPOPT_FPHSA_ARGS IPOPT_INCLUDE_DIR)
    find_path(IPOPT_INCLUDE_DIR NAMES IpIpoptNLP.hpp PATH_SUFFIXES coin coin-or)
endif()

if ("libipopt" IN_LIST IPOPT_FIND_COMPONENTS)
    # The libipopt component was requested.
    # The associated variable is IPOPT_LIBRARY.
    list(APPEND _IPOPT_FPHSA_ARGS IPOPT_LIBRARY)
    find_library(IPOPT_LIBRARY NAMES ipopt)
endif()

# Run the standard find_package() machinery.
find_package_handle_standard_args(IPOPT DEFAULT_MSG ${_IPOPT_FPHSA_ARGS})
unset(_IPOPT_FPHSA_ARGS)

if("header" IN_LIST IPOPT_FIND_COMPONENTS)
    mark_as_advanced(IPOPT_INCLUDE_DIR)

    if(IPOPT_FOUND AND NOT TARGET IPOPT::header)
        message(STATUS "Creating the 'IPOPT::header' imported target.")
        add_library(IPOPT::header INTERFACE IMPORTED)
        message(STATUS "Path to the ipopt headers: ${IPOPT_INCLUDE_DIR}")
        set_target_properties(IPOPT::header PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${IPOPT_INCLUDE_DIR}")
    endif()
endif()

if ("libipopt" IN_LIST IPOPT_FIND_COMPONENTS)
    mark_as_advanced(IPOPT_LIBRARY)

    if(IPOPT_FOUND AND NOT TARGET IPOPT::libipopt)
        message(STATUS "Creating the 'IPOPT::libipopt' imported target.")
        # Otherwise, we proceed as usual.
        message(STATUS "Path to libipopt: ${IPOPT_LIBRARY}")
        add_library(IPOPT::libipopt UNKNOWN IMPORTED)
        set_target_properties(IPOPT::libipopt PROPERTIES IMPORTED_LOCATION "${IPOPT_LIBRARY}")
    endif()
endif()
