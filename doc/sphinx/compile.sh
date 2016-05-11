#!/bin/sh
make clean
cd ../doxygen
doxygen > /dev/null
cp -r images/* xml
cd ../sphinx
make html > /dev/null
