#!/bin/bash

# Set OpenCL package to use
AMD=0
INTEL=1
NVIDIA=2
OPENCL_PACKAGE=$INTEL
#OPENCL_PACKAGE=$AMD

# Set compilation mode to use
RELEASE=0
DEBUG=1
COMPILATION=$RELEASE

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

# Set compilation flags
if [ "$COMPILATION" -eq "$RELEASE" ] ; then
    FLAGS="-O3 -DNDEBUG -m64 -fopenmp"
elif [ "$COMPILATION" -eq "$DEBUG" ] ; then
    FLAGS="-O0 -g -m64"
else
    echo "Unknown compilation mode"
fi

# Using g++
g++ -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/clBLASLinux ${FLAGS} -fPIC -c -o broccoli_lib.o broccoli_lib.cpp

# Make a library
ar rcs libBROCCOLI_LIB.a broccoli_lib.o

# Move to correct folder
if [ "$COMPILATION" -eq "$RELEASE" ] ; then
    mv libBROCCOLI_LIB.a ${BROCCOLI_GIT_DIRECTORY}/compiled/BROCCOLI_LIB/Linux/Release
elif [ "$COMPILATION" -eq "$DEBUG" ] ; then
    mv libBROCCOLI_LIB.a ${BROCCOLI_GIT_DIRECTORY}/compiled/BROCCOLI_LIB/Linux/Debug
else
    echo "Unknown compilation mode"
fi


# Old approach using nvcc from the CUDA toolkit
#nvcc -I${OPENCL_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen ${FLAGS} -m64 -Xcompiler -fPIC -c -o broccoli_lib.o broccoli_lib.cpp

