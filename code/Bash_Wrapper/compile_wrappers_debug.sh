#!/bin/bash

cp /home/andek/cuda-workspace/BROCCOLI_LIB/Debug/libBROCCOLI_LIB.a ../BROCCOLI_LIB

# Run make in nifticlib-2.0.0

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`
OPENCL_HEADER_DIRECTORY=/usr/local/cuda-5.0/include/CL
#OPENCL_HEADER_DIRECTORY=/opt/AMDAPP/include/CL # For AMD OpenCL package
OPENCL_LIBRARY_DIRECTORY=/usr/lib64

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
