cmake_minimum_required(VERSION 3.2)

project(udp_basic)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/yacma")

find_package(Pygmo REQUIRED)

include(YACMAPythonSetup)

YACMA_PYTHON_MODULE(udp_basic udp_basic.cpp)

target_link_libraries(udp_basic PRIVATE Pygmo::pygmo)
set_property(TARGET udp_basic PROPERTY CXX_STANDARD 11)
set_property(TARGET udp_basic PROPERTY CXX_STANDARD_REQUIRED YES)
set_property(TARGET udp_basic PROPERTY CXX_EXTENSIONS NO)

# Setup the installation path.
install(TARGETS udp_basic
 RUNTIME DESTINATION "${YACMA_PYTHON_MODULES_INSTALL_PATH}"
 LIBRARY DESTINATION "${YACMA_PYTHON_MODULES_INSTALL_PATH}"
)
