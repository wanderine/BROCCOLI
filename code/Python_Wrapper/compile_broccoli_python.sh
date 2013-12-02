#!/bin/bash

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`
BROCCOLI_LIB_DIRECTORY=${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB
OPENCL_DIRECTORY=/usr/include/CL #Containing opencl.h
PYTHON_DIRECTORY=/usr/include/python2.7

swig -c++ -python broccoli_lib.i

# Compile BROCCOLI
pushd ${BROCCOLI_LIB_DIRECTORY}
./compile_broccoli_library.sh
popd

gcc -fPIC -O3 -I${OPENCL_DIRECTORY} -I${PYTHON_DIRECTORY} -I${BROCCOLI_LIB_DIRECTORY} -I${BROCCOLI_LIB_DIRECTORY}/Eigen -o broccoli_lib_wrap.o -c broccoli_lib_wrap.cxx

# Make a library

g++ -fPIC -shared -o _broccoli_base.so -lOpenCL broccoli_lib_wrap.o ${BROCCOLI_LIB_DIRECTORY}/broccoli_lib.o
