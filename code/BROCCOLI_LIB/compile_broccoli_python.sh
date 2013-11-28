#!/bin/bash

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`
OPENCL_DIRECTORY=/usr/include/CL #Containing opencl.h
PYTHON_DIRECTORY=/usr/include/python2.7

swig -c++ -python broccoli_lib.i

# Compile BROCCOLI
nvcc -I${OPENCL_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -O3 -m64 -Xcompiler -fPIC -c -o broccoli_lib.o broccoli_lib.cpp
gcc -fPIC -I${OPENCL_DIRECTORY} -I${PYTHON_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -o broccoli_lib_wrap.o -c broccoli_lib_wrap.cxx

# Make a library

g++ -fPIC -shared -o _broccoli.so -lOpenCL broccoli_lib_wrap.o broccoli_lib.o
ar rcs libBROCCOLI_LIB.a broccoli_lib.o
