include(FindPackageHandleStandardArgs)

message(STATUS "Requested NLOPT components: ${NLOPT_FIND_COMPONENTS}")

# Check the components that were passed to find_package().
set(_NLOPT_ALLOWED_COMPONENTS header libnlopt)
foreach(_NLOPT_CUR_COMPONENT ${NLOPT_FIND_COMPONENTS})
    if(NOT ${_NLOPT_CUR_COMPONENT} IN_LIST _NLOPT_ALLOWED_COMPONENTS)
        message(FATAL_ERROR "'${_NLOPT_CUR_COMPONENT}' is not a valid component for NLOPT.")
    endif()
endforeach()
unset(_NLOPT_ALLOWED_COMPONENTS)

# Setup the list of arguments to be passed to
# find_package_handle_standard_args().
set(_NLOPT_FPHSA_ARGS)

if ("header" IN_LIST NLOPT_FIND_COMPONENTS)
    # The header component was requested.
    # The associated variable is NLOPT_INCLUDE_DIR.
    list(APPEND _NLOPT_FPHSA_ARGS NLOPT_INCLUDE_DIR)
    find_path(NLOPT_INCLUDE_DIR NAMES nlopt.h)
endif()

if ("libnlopt" IN_LIST NLOPT_FIND_COMPONENTS)
    # The libnlopt component was requested.
    # The associated variable is NLOPT_LIBRARY.
    list(APPEND _NLOPT_FPHSA_ARGS NLOPT_LIBRARY)
    find_library(NLOPT_LIBRARY NAMES nlopt)
endif()

# Run the standard find_package() machinery.
find_package_handle_standard_args(NLOPT DEFAULT_MSG ${_NLOPT_FPHSA_ARGS})
unset(_NLOPT_FPHSA_ARGS)

if("header" IN_LIST NLOPT_FIND_COMPONENTS)
    mark_as_advanced(NLOPT_INCLUDE_DIR)

    if(NLOPT_FOUND AND NOT TARGET NLOPT::header)
        message(STATUS "Creating the 'NLOPT::header' imported target.")
        add_library(NLOPT::header INTERFACE IMPORTED)
        message(STATUS "Path to the nlopt.h header: ${NLOPT_INCLUDE_DIR}")
        set_target_properties(NLOPT::header PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NLOPT_INCLUDE_DIR}")
    endif()
endif()

if ("libnlopt" IN_LIST NLOPT_FIND_COMPONENTS)
    mark_as_advanced(NLOPT_LIBRARY)

    if(NLOPT_FOUND AND NOT TARGET NLOPT::libnlopt)
        message(STATUS "Creating the 'NLOPT::libnlopt' imported target.")
        # Otherwise, we proceed as usual.
        message(STATUS "Path to libnlopt: ${NLOPT_LIBRARY}")
        add_library(NLOPT::libnlopt UNKNOWN IMPORTED)
        set_target_properties(NLOPT::libnlopt PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${NLOPT_LIBRARY}")
    endif()
endif()
