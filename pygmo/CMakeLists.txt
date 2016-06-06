# NOTE: this is already included from above, but keep it here
# in case one day we decide to split off Pyranha.
include(YACMAPythonSetup)

# Python version check.
if(${PYTHON_VERSION_MAJOR} LESS 2 OR (${PYTHON_VERSION_MAJOR} EQUAL 2 AND ${PYTHON_VERSION_MINOR} LESS 7))
	message(FATAL_ERROR "Minimum supported Python version is 2.7.")
endif()

if(NOT YACMA_NUMPY_INCLUDES_DIR)
	message(FATAL_ERROR "The NumPy headers could not be located.")
endif()

include_directories("${YACMA_NUMPY_INCLUDES_DIR}")

YACMA_PYTHON_MODULE(core core.cpp docstrings.cpp)
target_link_libraries(core ${MANDATORY_LIBRARIES} ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARIES})

# Setup the installation path.
set(PYGMO_INSTALL_PATH "${YACMA_PYTHON_MODULES_INSTALL_PATH}/pygmo")
install(TARGETS core
 RUNTIME DESTINATION ${PYGMO_INSTALL_PATH}
 LIBRARY DESTINATION ${PYGMO_INSTALL_PATH}
)

# Add the global init file.
install(FILES __init__.py DESTINATION ${PYGMO_INSTALL_PATH})
