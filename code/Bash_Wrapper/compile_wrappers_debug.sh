#!/bin/bash

AMD=0
INTEL=1
NVIDIA=2

# Set OpenCL package to use
OPENCL_PACKAGE=$INTEL

#cp /home/andek/cuda-workspace/BROCCOLI_LIB/Debug/libBROCCOLI_LIB.a ../BROCCOLI_LIB

# Run make in nifticlib-2.0.0

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`

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


g++ -g GetOpenCLInfo.cpp -I${OPENCL_HEADER_DIRECTORY} -L${OPENCL_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -lOpenCL -lBROCCOLI_LIB -o GetOpenCLInfo 

# Support for compressed files
g++ -g MotionCorrection.cpp -I${OPENCL_HEADER_DIRECTORY} -L${OPENCL_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lOpenCL -lBROCCOLI_LIB -lniftiio -lznz -lz -o MotionCorrection 

g++ -g RegisterTwoVolumes.cpp -I${OPENCL_HEADER_DIRECTORY} -L${OPENCL_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lOpenCL -lBROCCOLI_LIB -lniftiio -lznz -lz -o RegisterTwoVolumes

g++ -g TransformVolume.cpp -I${OPENCL_HEADER_DIRECTORY} -L${OPENCL_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lOpenCL -lBROCCOLI_LIB -lniftiio -lznz -lz -o TransformVolume

g++ -g RandomiseGroupLevel.cpp -I${OPENCL_HEADER_DIRECTORY} -L${OPENCL_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lOpenCL -lBROCCOLI_LIB -lniftiio -lznz -lz -o RandomiseGroupLevel

g++ -g FirstLevelAnalysis.cpp -I${OPENCL_HEADER_DIRECTORY} -L${OPENCL_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lOpenCL -lBROCCOLI_LIB -lniftiio -lznz -lz -o FirstLevelAnalysis

# gdb --args Program
# run

# Example debugging
# gdb --args ./FirstLevelAnalysis fMRI.nii T1_brain.nii MNI152_T1_1mm_brain.nii.gz regressors.txt  contrasts.txt -platform 2 -saveallaligned
# run + enter


