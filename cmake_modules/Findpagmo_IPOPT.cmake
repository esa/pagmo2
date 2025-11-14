include(FindPackageHandleStandardArgs)

if (NOT pagmo_IPOPT_FIND_QUIETLY)
  message(STATUS "Requested IPOPT components: ${pagmo_IPOPT_FIND_COMPONENTS}")
endif ()

# Check the components that were passed to find_package().
set(_pagmo_IPOPT_ALLOWED_COMPONENTS header libipopt)
foreach(_pagmo_IPOPT_CUR_COMPONENT ${pagmo_IPOPT_FIND_COMPONENTS})
    if(NOT "${_pagmo_IPOPT_CUR_COMPONENT}" IN_LIST _pagmo_IPOPT_ALLOWED_COMPONENTS)
        message(FATAL_ERROR "'${_pagmo_IPOPT_CUR_COMPONENT}' is not a valid component for IPOPT.")
    endif()
endforeach()
unset(_pagmo_IPOPT_ALLOWED_COMPONENTS)

# Setup the list of arguments to be passed to
# find_package_handle_standard_args().
set(_pagmo_IPOPT_FPHSA_ARGS)

if ("header" IN_LIST pagmo_IPOPT_FIND_COMPONENTS)
    # The header component was requested.
    # The associated variable is PAGMO_IPOPT_INCLUDE_DIR.
    list(APPEND _pagmo_IPOPT_FPHSA_ARGS PAGMO_IPOPT_INCLUDE_DIR)
    find_path(PAGMO_IPOPT_INCLUDE_DIR NAMES IpIpoptNLP.hpp PATH_SUFFIXES coin coin-or)
endif()

if ("libipopt" IN_LIST pagmo_IPOPT_FIND_COMPONENTS)
    # The libipopt component was requested.
    # The associated variable is PAGMO_IPOPT_LIBRARY.
    list(APPEND _pagmo_IPOPT_FPHSA_ARGS PAGMO_IPOPT_LIBRARY)
    find_library(PAGMO_IPOPT_LIBRARY NAMES ipopt ipopt-3)
endif()

# Run the standard find_package() machinery.
find_package_handle_standard_args(pagmo_IPOPT DEFAULT_MSG ${_pagmo_IPOPT_FPHSA_ARGS})
unset(_pagmo_IPOPT_FPHSA_ARGS)

if("header" IN_LIST pagmo_IPOPT_FIND_COMPONENTS)
    mark_as_advanced(PAGMO_IPOPT_INCLUDE_DIR)

    if(pagmo_IPOPT_FOUND AND NOT TARGET pagmo::IPOPT::header)
        if (NOT pagmo_IPOPT_FIND_QUIETLY)
          message(STATUS "Creating the 'pagmo::IPOPT::header' imported target.")
          message(STATUS "Path to the ipopt headers: ${PAGMO_IPOPT_INCLUDE_DIR}")
        endif ()
        add_library(pagmo::IPOPT::header INTERFACE IMPORTED)
        set_target_properties(pagmo::IPOPT::header PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PAGMO_IPOPT_INCLUDE_DIR}")
    endif()
endif()

if ("libipopt" IN_LIST pagmo_IPOPT_FIND_COMPONENTS)
    mark_as_advanced(PAGMO_IPOPT_LIBRARY)

    if(pagmo_IPOPT_FOUND AND NOT TARGET pagmo::IPOPT::libipopt)
        if (NOT pagmo_IPOPT_FIND_QUIETLY)
          message(STATUS "Creating the 'pagmo::IPOPT::libipopt' imported target.")
          message(STATUS "Path to libipopt: ${PAGMO_IPOPT_LIBRARY}")
        endif ()
        add_library(pagmo::IPOPT::libipopt UNKNOWN IMPORTED)
        set_target_properties(pagmo::IPOPT::libipopt PROPERTIES IMPORTED_LOCATION "${PAGMO_IPOPT_LIBRARY}")
    endif()
endif()
