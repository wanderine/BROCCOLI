#!/bin/bash

# Set compilation mode
RELEASE=0
DEBUG=1
COMPILATION=$RELEASE

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`

# Set directory containing opencl.h
OPENCL_HEADER_DIRECTORY=/System/Library/Frameworks/OpenCL.framework/Headers

# Compile BROCCOLI

if [ "$COMPILATION" -eq "$RELEASE" ] ; then
    FLAGS="-O3 -DNDEBUG"
elif [ "$COMPILATION" -eq "$DEBUG" ] ; then
    FLAGS="-O0 -g"
else
    echo "Unknown compilation mode"
fi


# Using g++
g++ -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen ${FLAGS} -m64 -fPIC -c -o broccoli_lib.o broccoli_lib.cpp

# Make a library
ar rcs libBROCCOLI_LIB.a broccoli_lib.o

# Move to correct folder
if [ "$COMPILATION" -eq "$RELEASE" ] ; then
    mv libBROCCOLI_LIB.a Compiled/Mac/Release
elif [ "$COMPILATION" -eq "$DEBUG" ] ; then
    mv libBROCCOLI_LIB.a Compiled/Mac/Debug
else
    echo "Unknown compilation mode"
fi


