#!/bin/bash

AMD=0
INTEL=1
NVIDIA=2

# Set OpenCL package to use
OPENCL_PACKAGE=$INTEL

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`

# Set directory containing opencl.h

# Need to install AMD OpenCL SDK first
if [ "$OPENCL_PACKAGE" -eq "$AMD" ] ; then
    OPENCL_HEADER_DIRECTORY1=/opt/AMDAPP/include 
    OPENCL_HEADER_DIRECTORY2=/opt/AMDAPP/include/CL 
# Need to install Intel OpenCL SDK first
elif [ "$OPENCL_PACKAGE" -eq "$INTEL" ] ; then
    OPENCL_HEADER_DIRECTORY1=/opt/intel/opencl-sdk/include 
    OPENCL_HEADER_DIRECTORY2=/opt/intel/opencl-sdk/include/CL
# Need to install Nvidia CUDA SDK first
elif [ "$OPENCL_PACKAGE" -eq "$NVIDIA" ] ; then
    OPENCL_HEADER_DIRECTORY1=/usr/include/CL 
    OPENCL_HEADER_DIRECTORY2=
else
    echo "-------------------------------------"
    echo "Unknown OpenCL package!"
    echo "-------------------------------------"
fi

# Compile BROCCOLI

DEBUG_FLAGS="-O0 -g"
RELEASE_FLAGS="-O3 -DNDEBUG"

FLAGS=${RELEASE_FLAGS}

# Using nvcc from the CUDA toolkit
#nvcc -I${OPENCL_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen ${FLAGS} -m64 -Xcompiler -fPIC -c -o broccoli_lib.o broccoli_lib.cpp

# Using g++
g++ -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen ${FLAGS} -m64 -fPIC -c -o broccoli_lib.o broccoli_lib.cpp

# Make a library
ar rcs libBROCCOLI_LIB.a broccoli_lib.o


