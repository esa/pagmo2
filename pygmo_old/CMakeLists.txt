include(YACMAPythonSetup)

YACMA_PYTHON_MODULE(_core _core.cpp)
target_link_libraries(_core ${MANDATORY_LIBRARIES} ${PYTHON_LIBRARIES})
