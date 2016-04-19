#!/bin/bash   
cd ../doxygen
doxygen Doxyfile
cd ../sphinx
make html

