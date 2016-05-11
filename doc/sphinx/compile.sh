#!/bin/sh   
make clean
cd ../doxygen
doxygen Doxyfile
cp -r images/* xml
cd ../sphinx
make html

