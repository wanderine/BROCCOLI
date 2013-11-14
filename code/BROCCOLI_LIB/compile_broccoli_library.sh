#!/bin/bash

BROCCOLI_GIT_DIRECTORY=/home/andek/Research_projects/BROCCOLI/BROCCOLI
OPENCL_DIRECTORY=/usr/local/cuda-5.0/include/CL #Containing opencl.h

# Compile BROCCOLI
nvcc -I${OPENCL_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -O3 -m64 -Xcompiler -fPIC -c -o broccoli_lib.o broccoli_lib.cpp

# Make a library
ar rcs libBROCCOLI_LIB.a broccoli_lib.o
