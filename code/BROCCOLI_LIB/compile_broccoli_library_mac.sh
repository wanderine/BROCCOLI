#!/bin/bash

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`

# Set directory containing opencl.h
OPENCL_HEADER_DIRECTORY=/System/Library/Frameworks/OpenCL.framework/Headers

# Compile BROCCOLI

DEBUG_FLAGS="-O0 -g"
RELEASE_FLAGS="-O3 -DNDEBUG"

FLAGS=${RELEASE_FLAGS}
#FLAGS=${DEBUG_FLAGS}

# Using g++
g++ -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen ${FLAGS} -m64 -fPIC -c -o broccoli_lib.o broccoli_lib.cpp

# Make a library
ar rcs libBROCCOLI_LIB.a broccoli_lib.o


