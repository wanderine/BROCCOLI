#!/bin/bash

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`

# Set OpenCL package to use
AMD=0
INTEL=1
NVIDIA=2
OPENCL_PACKAGE=$INTEL

# Set compilation mode to use
RELEASE=0
DEBUG=1
#COMPILATION=$RELEASE
COMPILATION=$DEBUG

# Fist run make for Nifti library
#cd nifticlib-2.0.0
#make
#cd ..

# Need to install AMD OpenCL SDK first
if [ "$OPENCL_PACKAGE" -eq "$AMD" ]; then
    OPENCL_HEADER_DIRECTORY1=/opt/AMDAPP/include 
    OPENCL_HEADER_DIRECTORY2=/opt/AMDAPP/include/CL 
    OPENCL_LIBRARY_DIRECTORY=/opt/AMDAPP/lib/x86_64 
# Need to install Intel OpenCL SDK and Intel OpenCL runtime first
elif [ "$OPENCL_PACKAGE" -eq "$INTEL" ]; then
    OPENCL_HEADER_DIRECTORY1=/opt/intel/opencl-sdk/include 
    OPENCL_HEADER_DIRECTORY2=/opt/intel/opencl-sdk/include/CL
    OPENCL_LIBRARY_DIRECTORY=/opt/intel/opencl/lib64
# Need to install Nvidia CUDA SDK first
elif [ "$OPENCL_PACKAGE" -eq "$NVIDIA" ]; then
    OPENCL_HEADER_DIRECTORY1=/usr/local/cuda-5.0/include/CL
    OPENCL_HEADER_DIRECTORY2=
    OPENCL_LIBRARY_DIRECTORY=/usr/lib64
else
    echo "Unknown OpenCL package!"
fi

# Set compilation flags
if [ "$COMPILATION" -eq "$RELEASE" ] ; then
    FLAGS="-O3 -DNDEBUG"
elif [ "$COMPILATION" -eq "$DEBUG" ] ; then
    FLAGS="-O0 -g"
else
    echo "Unknown compilation mode"
fi

# Support for compressed files
g++ RenderVolume.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lOpenCL -lniftiio -lznz -lz ${FLAGS} -lGLEW -lglut -o RenderVolume


