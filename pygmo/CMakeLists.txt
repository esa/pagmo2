include(YACMAPythonSetup)

# Python version check.
if(${PYTHON_VERSION_MAJOR} LESS 2 OR (${PYTHON_VERSION_MAJOR} EQUAL 2 AND ${PYTHON_VERSION_MINOR} LESS 7))
	message(FATAL_ERROR "Minimum supported Python version is 2.7.")
endif()

if(NOT YACMA_NUMPY_INCLUDES_DIR)
	message(FATAL_ERROR "The NumPy headers could not be located.")
endif()

include(PagmoFindBoost)

YACMA_PYTHON_MODULE(core core.cpp docstrings.cpp)
target_link_libraries(core PRIVATE Boost::boost Boost::python Boost::disable_autolinking pagmo)
target_include_directories(core SYSTEM PRIVATE "${YACMA_NUMPY_INCLUDES_DIR}")
target_compile_options(core PRIVATE "$<$<CONFIG:DEBUG>:${PAGMO_CXX_FLAGS_DEBUG}>" "$<$<CONFIG:RELEASE>:${PAGMO_CXX_FLAGS_RELEASE}>")
set_property(TARGET core PROPERTY CXX_STANDARD 11)
set_property(TARGET core PROPERTY CXX_STANDARD_REQUIRED YES)
set_property(TARGET core PROPERTY CXX_EXTENSIONS NO)

# Setup the installation path.
set(PYGMO_INSTALL_PATH "${YACMA_PYTHON_MODULES_INSTALL_PATH}/pygmo")
install(TARGETS core
 RUNTIME DESTINATION ${PYGMO_INSTALL_PATH}
 LIBRARY DESTINATION ${PYGMO_INSTALL_PATH}
)

# Add submodules directories
ADD_SUBDIRECTORY(plotting)

# Add the Python files.
install(FILES __init__.py test.py DESTINATION ${PYGMO_INSTALL_PATH})
