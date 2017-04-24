if(IPOPT_INCLUDE_DIR AND IPOPT_LIBRARY)
	# Already in cache, be silent
	set(IPOPT_FIND_QUIETLY TRUE)
endif()

find_path(IPOPT_INCLUDE_DIR NAMES IpIpoptNLP.hpp PATH_SUFFIXES coin)
find_library(IPOPT_LIBRARY NAMES ipopt)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(IPOPT DEFAULT_MSG IPOPT_INCLUDE_DIR IPOPT_LIBRARY)

mark_as_advanced(IPOPT_INCLUDE_DIR IPOPT_LIBRARY)

# NOTE: this has been adapted from CMake's FindPNG.cmake.
if(IPOPT_FOUND AND NOT TARGET IPOPT::ipopt)
	add_library(IPOPT::ipopt UNKNOWN IMPORTED)
    set_target_properties(IPOPT::ipopt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${IPOPT_INCLUDE_DIR}")
    set_target_properties(IPOPT::ipopt PROPERTIES
        IMPORTED_LOCATION "${IPOPT_LIBRARY}")
endif()
