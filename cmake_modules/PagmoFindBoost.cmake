set(_PAGMO_REQUIRED_BOOST_LIBS)
if(PAGMO_BUILD_PYGMO)
    list(APPEND _PAGMO_REQUIRED_BOOST_LIBS python)
endif()
message(STATUS "Required Boost libraries: ${_PAGMO_REQUIRED_BOOST_LIBS}")
find_package(Boost 1.55.0 REQUIRED COMPONENTS "${_PAGMO_REQUIRED_BOOST_LIBS}")
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Not all requested Boost components were found, exiting.")
endif()
message(STATUS "Detected Boost version: ${Boost_VERSION}")
message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
if(NOT _Boost_IMPORTED_TARGETS)
    # NOTE: CMake's Boost finding module will not provide imported targets for recent Boost versions, as it needs
    # an explicit mapping specifying the dependencies between the various Boost libs (and this is version-dependent).
    # If we are here, it means that Boost was correctly found with all the needed components, but the Boost version
    # found is too recent and imported targets are not available. We will reconstruct them here in order to be able
    # to link to targets rather than using the variables defined by the FindBoost.cmake module.
    message(STATUS "The imported Boost targets are not available, creating them manually.")
    if(NOT TARGET Boost::boost)
        add_library(Boost::boost INTERFACE IMPORTED)
        set_target_properties(Boost::boost PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
    endif()
    if(NOT TARGET Boost::disable_autolinking)
        add_library(Boost::disable_autolinking INTERFACE IMPORTED)
        if(WIN32)
            set_target_properties(Boost::disable_autolinking PROPERTIES INTERFACE_COMPILE_DEFINITIONS "BOOST_ALL_NO_LIB")
        endif()
    endif()
    foreach(_PAGMO_BOOST_COMPONENT ${_PAGMO_REQUIRED_BOOST_LIBS})
        message(STATUS "Creating the 'Boost::${_PAGMO_BOOST_COMPONENT}' imported target.")
        string(TOUPPER ${_PAGMO_BOOST_COMPONENT} _PAGMO_BOOST_UPPER_COMPONENT)
        if(Boost_USE_STATIC_LIBS)
            add_library(Boost::${_PAGMO_BOOST_COMPONENT} STATIC IMPORTED)
        else()
            add_library(Boost::${_PAGMO_BOOST_COMPONENT} UNKNOWN IMPORTED)
        endif()
        set_target_properties(Boost::${_PAGMO_BOOST_COMPONENT} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
        set_target_properties(Boost::${_PAGMO_BOOST_COMPONENT} PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
            IMPORTED_LOCATION "${Boost_${_PAGMO_BOOST_UPPER_COMPONENT}_LIBRARY}")
    endforeach()
    # NOTE: the FindBoost macro also sets the Release and Debug counterparts of the properties above.
    # It seems like it is not necessary for our own uses, but keep it in mind for the future.
endif()
