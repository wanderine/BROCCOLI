#!/bin/bash

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`
OPENCL_DIRECTORY=/usr/include/CL #Containing opencl.h

# Compile BROCCOLI

DEBUG_FLAGS="-O0 -g"
RELEASE_FLAGS="-O3 -DNDEBUG"

FLAGS=${DEBUG_FLAGS}

nvcc -I${OPENCL_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen ${FLAGS} -m64 -Xcompiler -fPIC -c -o broccoli_lib.o broccoli_lib.cpp

# Make a library
ar rcs libBROCCOLI_LIB.a broccoli_lib.o
