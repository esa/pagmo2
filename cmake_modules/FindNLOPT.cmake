if(NLOPT_INCLUDE_DIR AND NLOPT_LIBRARY)
	# Already in cache, be silent
	set(NLOPT_FIND_QUIETLY TRUE)
endif()

find_path(NLOPT_INCLUDE_DIR NAMES nlopt.h)
find_library(NLOPT_LIBRARY NAMES nlopt)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(NLOPT DEFAULT_MSG NLOPT_INCLUDE_DIR NLOPT_LIBRARY)

mark_as_advanced(NLOPT_INCLUDE_DIR NLOPT_LIBRARY)

# NOTE: this has been adapted from CMake's FindPNG.cmake.
if(NLOPT_FOUND AND NOT TARGET NLOPT::nlopt)
    add_library(NLOPT::nlopt UNKNOWN IMPORTED)
    set_target_properties(NLOPT::nlopt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NLOPT_INCLUDE_DIR}")
    set_target_properties(NLOPT::nlopt PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${NLOPT_LIBRARY}")
endif()
